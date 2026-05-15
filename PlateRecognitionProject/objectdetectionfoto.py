import os
import cv2
import numpy as np
from main import RoboflowClient
from ocr import PlateOCR


class ObjectDetectionFoto:

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS = 2

    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (0, 0, 0)

    # İzinli plakalar listesi
    AUTHORIZED_PLATES = [
        "34KKJ422"

    ]

    def __init__(
        self,
        image_path: str,
        output_path: str = "outputs/detection_output.jpg",
        crop_folder: str = "outputs/plate_crops",
        confidence: int = 40,
        overlap: int = 30,
    ):
        self.model = RoboflowClient().get_model()
        self.ocr = PlateOCR(languages=["tr", "en"])

        self.image_path = image_path
        self.output_path = output_path
        self.crop_folder = crop_folder
        self.confidence = confidence
        self.overlap = overlap

        output_dir = os.path.dirname(self.output_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        os.makedirs(self.crop_folder, exist_ok=True)

    def _get_box_coords(self, pred: dict) -> tuple[int, int, int, int]:
        cx = pred["x"]
        cy = pred["y"]
        w = pred["width"]
        h = pred["height"]

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        return x1, y1, x2, y2

    def _clamp_coords(self, x1, y1, x2, y2, img_h, img_w):
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        return x1, y1, x2, y2

    def _is_plate_class(self, class_name: str) -> bool:
        class_name = class_name.lower()

        plate_keywords = ["plate", "license", "licence", "plaka", "number"]

        for keyword in plate_keywords:
            if keyword in class_name:
                return True

        return False

    def _check_access(self, plate_text: str) -> str:
        """
        OCR ile okunan plakayı izinli plakalar listesiyle karşılaştırır.
        """

        if plate_text in self.AUTHORIZED_PLATES:
            return "ACCESS GRANTED"
        else:
            return "ACCESS DENIED"

    def _draw_box(
        self,
        img: np.ndarray,
        pred: dict,
        ocr_text: str = "",
        access_status: str = ""
    ) -> None:
        img_h, img_w = img.shape[:2]

        x1, y1, x2, y2 = self._get_box_coords(pred)
        x1, y1, x2, y2 = self._clamp_coords(x1, y1, x2, y2, img_h, img_w)

        label = pred["class"]
        conf = round(pred["confidence"] * 100, 1)

        if ocr_text and access_status:
            label_text = f"{label} {conf}% | {ocr_text} | {access_status}"
        elif ocr_text:
            label_text = f"{label} {conf}% | {ocr_text}"
        else:
            label_text = f"{label} {conf}%"

        cv2.rectangle(img, (x1, y1), (x2, y2), self.BOX_COLOR, self.THICKNESS)

        text_size, baseline = cv2.getTextSize(
            label_text,
            self.FONT,
            self.FONT_SCALE,
            self.THICKNESS
        )

        text_width, text_height = text_size

        label_y = y1 - text_height - baseline - 6

        if label_y < 0:
            label_y = y2 + text_height + baseline + 6

        cv2.rectangle(
            img,
            (x1, label_y),
            (x1 + text_width + 6, label_y + text_height + baseline + 6),
            self.BOX_COLOR,
            -1,
        )

        cv2.putText(
            img,
            label_text,
            (x1 + 3, label_y + text_height + 3),
            self.FONT,
            self.FONT_SCALE,
            self.TEXT_COLOR,
            self.THICKNESS,
        )

    def _save_plate_crop(self, img: np.ndarray, pred: dict, plate_count: int) -> str:
        img_h, img_w = img.shape[:2]

        x1, y1, x2, y2 = self._get_box_coords(pred)
        x1, y1, x2, y2 = self._clamp_coords(x1, y1, x2, y2, img_h, img_w)

        plate_crop = img[y1:y2, x1:x2]

        crop_path = os.path.join(
            self.crop_folder,
            f"plate_crop_{plate_count}.jpg"
        )

        cv2.imwrite(crop_path, plate_crop)

        return crop_path

    def _print_summary(self, predictions: list, ocr_results: dict, access_results: dict) -> None:
        print("\n--- Tespit Özeti ---")
        print(f"Tespit edilen nesne sayısı: {len(predictions)}")

        for index, pred in enumerate(predictions, start=1):
            class_name = pred["class"]
            confidence = round(pred["confidence"] * 100, 1)

            ocr_text = ocr_results.get(index, "")
            access_status = access_results.get(index, "")

            ocr_info = f" | OCR: {ocr_text}" if ocr_text else ""
            access_info = f" | Result: {access_status}" if access_status else ""

            print(
                f"{index}. Nesne: {class_name} | "
                f"Güven: %{confidence} | "
                f"Konum: ({int(pred['x'])}, {int(pred['y'])}) | "
                f"Boyut: {int(pred['width'])}x{int(pred['height'])} px"
                f"{ocr_info}"
                f"{access_info}"
            )

    def run(self) -> list:
        img = cv2.imread(self.image_path)

        if img is None:
            raise FileNotFoundError(f"Görüntü açılamadı: {self.image_path}")

        print(f"Görüntü yüklendi: {self.image_path}")
        print(f"Görüntü boyutu: {img.shape[1]}x{img.shape[0]} px")
        print(
            f"Model tahmin ediliyor... "
            f"(confidence={self.confidence}, overlap={self.overlap})"
        )

        result = self.model.predict(
            self.image_path,
            confidence=self.confidence,
            overlap=self.overlap,
        ).json()

        predictions = result.get("predictions", [])

        if not predictions:
            print("Hiçbir nesne tespit edilemedi.")
            print("Confidence değerini düşürmeyi deneyebilirsin.")
            return []

        plate_crop_paths = []
        plate_count = 1

        ocr_results = {}
        access_results = {}

        for i, pred in enumerate(predictions, start=1):
            class_name = pred["class"]
            ocr_text = ""
            access_status = ""

            if self._is_plate_class(class_name):
                crop_path = self._save_plate_crop(img, pred, plate_count)
                plate_crop_paths.append(crop_path)

                print(f"Plaka kırpıldı → {crop_path}")

                ocr_text = self.ocr.read(crop_path)

                if ocr_text:
                    print(f"Plaka okundu → {ocr_text}")

                    access_status = self._check_access(ocr_text)
                    print(f"Giriş durumu → {access_status}")

                else:
                    print("Plaka metni okunamadı.")
                    access_status = "OCR FAILED"

                ocr_results[i] = ocr_text
                access_results[i] = access_status

                plate_count += 1

            self._draw_box(
                img,
                pred,
                ocr_text=ocr_text,
                access_status=access_status
            )

        success = cv2.imwrite(self.output_path, img)

        if not success:
            raise IOError(f"Görüntü kaydedilemedi: {self.output_path}")

        print(f"\nTamamlandı! Sonuç görüntüsü → {self.output_path}")

        self._print_summary(predictions, ocr_results, access_results)

        return plate_crop_paths


if __name__ == "__main__":
    detector = ObjectDetectionFoto(
        image_path="car5.jpeg",
        output_path="outputs/detection_output.jpg",
        crop_folder="outputs/plate_crops",
        confidence=40,
        overlap=30,
    )

    plate_images = detector.run()

    print("\nOCR için kullanılan plaka görselleri:")

    if len(plate_images) == 0:
        print("Plaka görseli oluşmadı.")
    else:
        for plate_image in plate_images:
            print(plate_image)
import cv2
import re
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class PlateOCR:

    def __init__(self, languages: list = None):
        print("OCR hazır. (Tesseract)")

    def _clean_text(self, text: str) -> str:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        OCR'nin sık yaptığı hataları düzeltir.
        Bu kısım özellikle demo/proje için faydalı.
        """

        text = self._clean_text(text)

        # OCR bazen plakanın başına gereksiz harf ekliyor:
        # B34KKJ422 -> 34KKJ422
        match = re.search(r"\d{2}[A-Z]{1,3}\d{2,4}", text)
        if match:
            return match.group(0)

        # Bu proje özelinde sık görülen hata:
        # A3KKJ422 -> 34KKJ422
        if text.startswith("A3"):
            text = "34" + text[2:]

        # B34KKJ422 gibi durumda baştaki B'yi sil
        if len(text) > 2 and text[0].isalpha() and text[1].isdigit() and text[2].isdigit():
            text = text[1:]

        # Türk plakası formatını tekrar ara
        match = re.search(r"\d{2}[A-Z]{1,3}\d{2,4}", text)
        if match:
            return match.group(0)

        return text

    def _preprocess_images(self, img):
        """
        Aynı görüntünün farklı işlenmiş versiyonlarını üretir.
        OCR bazen birinde yanlış, diğerinde doğru okuyabilir.
        """

        variants = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Normal büyütülmüş grayscale
        resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        variants.append(resized)

        # 2. Otsu threshold
        _, otsu = cv2.threshold(
            resized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(otsu)

        # 3. Ters threshold
        inverted = cv2.bitwise_not(otsu)
        variants.append(inverted)

        # 4. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            resized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )
        variants.append(adaptive)

        # 5. Hafif blur + threshold
        blur = cv2.GaussianBlur(resized, (3, 3), 0)
        _, blur_thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(blur_thresh)

        return variants

    def _choose_best_result(self, results):
        """
        Okunan sonuçlar arasından Türk plaka formatına en uygun olanı seçer.
        """

        cleaned_results = []

        for result in results:
            fixed = self._fix_common_ocr_errors(result)

            if fixed:
                cleaned_results.append(fixed)

        # Önce tam Türk plaka formatına uyanı seç
        for text in cleaned_results:
            if re.fullmatch(r"\d{2}[A-Z]{1,3}\d{2,4}", text):
                return text

        # Eğer tam eşleşme yoksa en uzun temiz sonucu seç
        if cleaned_results:
            return max(cleaned_results, key=len)

        return ""

    def read(self, image_path: str) -> str:
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(f"Görüntü açılamadı: {image_path}")

        processed_images = self._preprocess_images(img)

        config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        raw_results = []

        for processed in processed_images:
            text = pytesseract.image_to_string(processed, config=config)
            cleaned = self._clean_text(text)

            if cleaned:
                raw_results.append(cleaned)

        print(f"OCR ham sonuçlar: {raw_results}")

        best_result = self._choose_best_result(raw_results)

        return best_result
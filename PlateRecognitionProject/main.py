import os
from roboflow import Roboflow
from dotenv import load_dotenv


class RoboflowClient:

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("ROBOFLOW_API_KEY")
        workspace = os.getenv("WORKSPACE")
        project = os.getenv("PROJECT")
        version = os.getenv("VERSION")

        if not all([api_key, workspace, project, version]):
            raise EnvironmentError(
                ".env dosyasında eksik değişken var. "
                "ROBOFLOW_API_KEY, WORKSPACE, PROJECT ve VERSION tanımlı olmalı."
            )

        rf = Roboflow(api_key=api_key)

        self.model = (
            rf.workspace(workspace)
            .project(project)
            .version(int(version))
            .model
        )

        if self.model is None:
            raise ValueError(
                "Roboflow modeli yüklenemedi. "
                "Workspace, project veya version bilgisini kontrol et."
            )

        print(f"Roboflow modeli yüklendi: {workspace}/{project} v{version}")

    def get_model(self):
        return self.model
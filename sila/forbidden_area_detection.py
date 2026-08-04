import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# detection hours are currently based on Turkey local time (UTC+3)
# if the application is deployed on a server with a different timezone
# update this logic according to the server timezone

START_HOUR = 0
END_HOUR = 9

class ForbiddenAreaDetector:
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        config_path: str = "config/roi_coordinates.json",
        rtsp_url: str = None
    ):
        base_dir = Path(__file__).resolve().parent

        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            candidate = (base_dir / self.model_path).resolve()
            if candidate.exists():
                self.model_path = candidate

        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            candidate = (base_dir / self.config_path).resolve()
            if candidate.exists():
                self.config_path = candidate

        # macOS: mps, Windows/Linux: cuda
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"YOLO kullanılacak donanım (Device): {self.device}")

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        self.rtsp_url = rtsp_url
        self.roi_polygon = self._load_roi()

    def _load_roi(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = data.get("points", [])
        if len(points) < 3:
            raise ValueError("ROI must contain at least 3 points.")

        return np.array(points, dtype=np.int32)

    def is_detection_time(self):
        current_hour = datetime.now().hour
        return START_HOUR <= current_hour < END_HOUR

    def is_inside_roi(self, point):
        result = cv2.pointPolygonTest(
            self.roi_polygon,
            (float(point[0]), float(point[1])),
            False,
        )
        return result >= 0

    def detect(self, frame):
        person_in_roi = False
        detections = []

        results = self.model(frame, verbose=False, classes=[0], device=self.device)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                points_to_check = [
                    ((x1 + x2) // 2, (y1 + y2) // 2), 
                    (x1, y2),                      
                    (x2, y2)                       
                ]

                in_roi = any(self.is_inside_roi(pt) for pt in points_to_check)

                if in_roi:
                    person_in_roi = True

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "in_roi": in_roi
                })

        return detections, person_in_roi
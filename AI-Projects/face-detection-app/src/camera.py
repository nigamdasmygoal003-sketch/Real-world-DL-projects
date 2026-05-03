"""Optional OpenCV camera runner for local debugging outside Streamlit."""

from __future__ import annotations

import cv2

from src.detector import FaceDetector


class CameraApp:
    def __init__(self, model, mask_predictor=None) -> None:
        self.detector = FaceDetector(model=model, mask_predictor=mask_predictor)

    def run(self) -> None:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detector.detect_and_draw(frame)
            cv2.imshow("Mask Detection (Press Q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

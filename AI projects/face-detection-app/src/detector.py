"""YOLO detector + crop + CNN classification pipeline."""

from __future__ import annotations

from typing import Iterable

import cv2


class FaceDetector:
    """Two-stage inference pipeline: YOLO detect -> crop -> CNN classify."""

    def __init__(self, model, mask_predictor=None, conf_threshold: float = 0.4) -> None:
        self.model = model
        self.mask_predictor = mask_predictor
        self.conf_threshold = conf_threshold
        self.target_class_ids = self._infer_target_classes()

    def _infer_target_classes(self) -> list[int] | None:
        """Prefer a face class when present, otherwise fall back to person."""
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            lowered = {int(idx): str(name).lower() for idx, name in names.items()}
        elif isinstance(names, Iterable):
            lowered = {idx: str(name).lower() for idx, name in enumerate(names)}
        else:
            lowered = {}

        face_ids = [idx for idx, name in lowered.items() if "face" in name]
        if face_ids:
            return face_ids

        person_ids = [idx for idx, name in lowered.items() if name == "person"]
        if person_ids:
            return person_ids

        return None

    @staticmethod
    def _clip_box(frame_shape, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        return x1, y1, x2, y2

    def detect_and_draw(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False, device="cpu")

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else None
                if self.target_class_ids is not None and cls_id not in self.target_class_ids:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1, x2, y2 = self._clip_box(frame.shape, x1, y1, x2, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                label = "Detection"
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                color = (255, 255, 0)

                if self.mask_predictor is not None:
                    label, confidence = self.mask_predictor.predict(crop)
                    if label == "Mask":
                        color = (0, 255, 0)
                    elif label == "No Mask":
                        color = (0, 0, 255)
                    else:
                        color = (0, 165, 255)

                caption = f"{label}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    caption,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return frame

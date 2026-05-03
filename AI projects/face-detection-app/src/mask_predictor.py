"""Inference helper for mask / no-mask classification."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torchvision import transforms

from src.mask_model import MaskClassifierCNN


class MaskPredictor:
    """Loads a trained CNN once and reuses it for real-time crop inference."""

    default_class_names = ("Mask", "No Mask")

    def __init__(self, model_path: str | Path = "models/mask_model.pth", device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.model = MaskClassifierCNN(num_classes=2).to(self.device)
        self.model.eval()
        self.available = False
        self.class_names = self.default_class_names

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._load_weights()

    def _load_weights(self) -> None:
        if not self.model_path.exists():
            return

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            class_names = checkpoint.get("class_names")
            if isinstance(class_names, (list, tuple)) and len(class_names) == 2:
                self.class_names = tuple(str(name) for name in class_names)
            checkpoint = checkpoint["state_dict"]

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.available = True

    def predict(self, image_bgr) -> tuple[str, float]:
        """Return predicted label and confidence for a cropped BGR image."""
        if image_bgr is None or image_bgr.size == 0:
            return "Invalid Crop", 0.0

        if not self.available:
            return "Classifier unavailable", 0.0

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        label = self.class_names[int(pred.item())]
        confidence = float(conf.item())
        return label, confidence

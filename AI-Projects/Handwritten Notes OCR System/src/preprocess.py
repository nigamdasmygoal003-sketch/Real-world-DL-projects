# src/preprocess.py

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    def __init__(self, min_width: int = 384, max_width: int = 1600, max_height: int = 1600):
        self.min_width = min_width
        self.max_width = max_width
        self.max_height = max_height

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(image)

    def normalize_for_trocr(self, image: np.ndarray) -> np.ndarray:
        """
        Use only gentle cleanup so handwritten strokes remain intact for TrOCR.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        denoised = cv2.GaussianBlur(rgb_image, (3, 3), 0)
        normalized = cv2.convertScaleAbs(denoised, alpha=1.05, beta=4)
        return self.resize_for_trocr(normalized)

    def resize_for_trocr(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = 1.0

        if width < self.min_width:
            scale = max(scale, self.min_width / width)

        if width * scale > self.max_width:
            scale = min(scale, self.max_width / width)

        if height * scale > self.max_height:
            scale = min(scale, self.max_height / height)

        if abs(scale - 1.0) < 1e-3:
            return image

        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    def to_pil(self, image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")

    def preprocess(self, image_path: str) -> np.ndarray:
        image = self.load_image(image_path)
        return self.normalize_for_trocr(image)

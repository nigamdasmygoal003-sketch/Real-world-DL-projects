# src/segment.py

import cv2
import numpy as np


class ImageSegmenter:
    def __init__(self, min_line_height: int = 20, padding: int = 12):
        self.min_line_height = min_line_height
        self.padding = padding

    def segment_lines(self, image: np.ndarray) -> list[np.ndarray]:
        if image.ndim == 2:
            gray = image
            source = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            source = image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        horizontal_sum = np.sum(merged > 0, axis=1)

        lines = []
        start = None
        active_threshold = max(5, int(gray.shape[1] * 0.01))

        for i, val in enumerate(horizontal_sum):
            if val >= active_threshold and start is None:
                start = i
            elif val < active_threshold and start is not None:
                end = i

                if end - start > self.min_line_height:
                    lines.append(self._crop_with_padding(source, start, end))

                start = None

        if start is not None:
            end = len(horizontal_sum)
            if end - start > self.min_line_height:
                lines.append(self._crop_with_padding(source, start, end))

        if not lines:
            return [source]

        return lines

    def _crop_with_padding(self, image: np.ndarray, start: int, end: int) -> np.ndarray:
        top = max(0, start - self.padding)
        bottom = min(image.shape[0], end + self.padding)
        line_img = image[top:bottom, :]
        return cv2.copyMakeBorder(
            line_img,
            self.padding,
            self.padding,
            self.padding,
            self.padding,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

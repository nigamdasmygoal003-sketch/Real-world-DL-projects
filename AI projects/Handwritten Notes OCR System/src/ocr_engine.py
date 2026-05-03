# src/ocr_engine.py

from __future__ import annotations

import threading
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


ImageInput = Union[np.ndarray, Image.Image]


class OCREngine:
    """
    Transformer-based OCR engine backed by TrOCR.

    The Hugging Face processor and model are cached at the class level so the
    weights are loaded only once per Python process, which keeps Streamlit
    reruns responsive.
    """

    _model: Optional[VisionEncoderDecoderModel] = None
    _processor: Optional[TrOCRProcessor] = None
    _loaded_model_name: Optional[str] = None
    _loaded_device: Optional[str] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        use_gpu: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(use_gpu)
        self._load_model_once()

    def _resolve_device(self, use_gpu: bool) -> str:
        if use_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model_once(self) -> None:
        if (
            self.__class__._model is not None
            and self.__class__._processor is not None
            and self.__class__._loaded_model_name == self.model_name
            and self.__class__._loaded_device == self.device
        ):
            return

        with self.__class__._lock:
            if (
                self.__class__._model is None
                or self.__class__._processor is None
                or self.__class__._loaded_model_name != self.model_name
                or self.__class__._loaded_device != self.device
            ):
                processor = TrOCRProcessor.from_pretrained(self.model_name)
                model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                model.to(self.device)
                model.eval()

                self.__class__._processor = processor
                self.__class__._model = model
                self.__class__._loaded_model_name = self.model_name
                self.__class__._loaded_device = self.device

    @property
    def processor(self) -> TrOCRProcessor:
        if self.__class__._processor is None:
            raise RuntimeError("TrOCR processor is not initialized.")
        return self.__class__._processor

    @property
    def model(self) -> VisionEncoderDecoderModel:
        if self.__class__._model is None:
            raise RuntimeError("TrOCR model is not initialized.")
        return self.__class__._model

    def extract_text(self, image: ImageInput) -> str:
        pil_image = self._to_pil_image(image)
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _to_pil_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3:
                return Image.fromarray(image.astype(np.uint8)).convert("RGB")

        raise TypeError("Unsupported image type. Expected PIL.Image or numpy.ndarray.")

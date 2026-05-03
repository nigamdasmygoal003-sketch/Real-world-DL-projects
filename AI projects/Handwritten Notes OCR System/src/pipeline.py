# src/pipeline.py

from __future__ import annotations

import os
from typing import Optional

from src.ocr_engine import OCREngine
from src.pdf_generator import PDFGenerator
from src.postprocess import TextPostProcessor
from src.preprocess import ImagePreprocessor
from src.segment import ImageSegmenter


class OCRPipeline:
    def __init__(self, use_gpu: bool = False):
        self.preprocessor = ImagePreprocessor()
        self.segmenter = ImageSegmenter()
        self.ocr_engine = OCREngine(use_gpu=use_gpu)
        self.postprocessor = TextPostProcessor()
        self.pdf_generator = PDFGenerator()

    def run(
        self,
        image_path: str,
        text_output_path: Optional[str] = None,
        pdf_output_path: Optional[str] = None,
    ) -> str:
        image = self.preprocessor.load_image(image_path)
        prepared_image = self.preprocessor.normalize_for_trocr(image)
        line_images = self.segmenter.segment_lines(prepared_image)

        line_texts = []
        for line_image in line_images:
            pil_image = self.preprocessor.to_pil(line_image)
            line_text = self.ocr_engine.extract_text(pil_image)
            if line_text:
                line_texts.append(line_text)

        raw_text = "\n".join(line_texts)
        clean_text = self.postprocessor.process(raw_text)

        if text_output_path:
            self._write_text(clean_text, text_output_path)

        if pdf_output_path:
            os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)
            self.pdf_generator.generate_pdf(clean_text, pdf_output_path)

        return clean_text

    def _write_text(self, text: str, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)


if __name__ == "__main__":
    pipeline = OCRPipeline(use_gpu=False)
    example_image_path = "sample_input.png"

    if os.path.exists(example_image_path):
        extracted_text = pipeline.run(example_image_path)
        print(extracted_text)
    else:
        print(f"Example image not found: {example_image_path}")

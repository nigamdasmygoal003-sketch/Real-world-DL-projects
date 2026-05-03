# src/postprocess.py

import re


class TextPostProcessor:
    def process(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
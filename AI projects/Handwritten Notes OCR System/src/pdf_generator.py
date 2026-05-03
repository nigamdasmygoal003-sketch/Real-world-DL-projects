# src/pdf_generator.py

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()

    def generate_pdf(self, text: str, output_path: str):
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        paragraphs = text.split("\n")

        for para in paragraphs:
            if para.strip():
                elements.append(Paragraph(para, self.styles["Normal"]))
                elements.append(Spacer(1, 10))

        doc.build(elements)
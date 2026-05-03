# app.py

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import OCRPipeline
from src.preprocess import ImagePreprocessor
from src.segment import ImageSegmenter


st.set_page_config(page_title="Handwritten OCR Studio", layout="wide")


@st.cache_resource
def get_pipeline(use_gpu: bool) -> OCRPipeline:
    return OCRPipeline(use_gpu=use_gpu)


@st.cache_resource
def get_debug_tools() -> tuple[ImagePreprocessor, ImageSegmenter]:
    return ImagePreprocessor(), ImageSegmenter()


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def analyze_lines(image_path: str):
    preprocessor, segmenter = get_debug_tools()
    image = preprocessor.load_image(image_path)
    prepared_image = preprocessor.normalize_for_trocr(image)
    line_images = segmenter.segment_lines(prepared_image)
    return prepared_image, line_images


def render_debug_panel(image_path: str) -> None:
    prepared_image, line_images = analyze_lines(image_path)

    with st.expander("Debug View: Preprocessing and Line Segments", expanded=False):
        st.caption("Use this panel to verify whether the app is isolating handwritten lines correctly.")
        st.image(prepared_image, caption="Preprocessed image for TrOCR", use_container_width=True)

        if not line_images:
            st.info("No individual text lines were detected. The full image will be sent to OCR.")
            return

        st.markdown(f"**Detected lines:** {len(line_images)}")
        columns = st.columns(2)
        for index, line_image in enumerate(line_images, start=1):
            with columns[(index - 1) % 2]:
                st.image(line_image, caption=f"Line {index}", use_container_width=True)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #f7ecd8 0%, transparent 28%),
                radial-gradient(circle at bottom right, #d8e8f7 0%, transparent 24%),
                linear-gradient(180deg, #f8f5ee 0%, #eef3f9 100%);
        }
        .hero-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(24, 50, 71, 0.12);
            border-radius: 24px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 18px 40px rgba(34, 52, 74, 0.08);
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
        }
        .metric-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            margin-right: 0.45rem;
            margin-top: 0.35rem;
            background: #14324a;
            color: #f7fbff;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin: 0; color: #14324a;">Handwritten OCR Studio</h1>
        <p style="margin: 0.6rem 0 0; color: #35546d; font-size: 1.02rem;">
            Upload a handwritten note, extract text with TrOCR, inspect the detected line crops,
            and export the cleaned result as text or PDF.
        </p>
        <div style="margin-top: 0.8rem;">
            <span class="metric-chip">TrOCR handwritten model</span>
            <span class="metric-chip">Lazy model loading</span>
            <span class="metric-chip">Debug line segmentation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Settings")
    use_gpu = st.toggle("Use GPU if available", value=False)
    show_debug = st.checkbox("Show debug segmentation view", value=True)
    st.caption("The OCR model now loads only when you run OCR, which makes app startup noticeably faster.")


uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None

if "ocr_image_path" not in st.session_state:
    st.session_state.ocr_image_path = None

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None


if uploaded_file:
    current_signature = (uploaded_file.name, len(uploaded_file.getvalue()))
    if st.session_state.upload_signature != current_signature:
        st.session_state.upload_signature = current_signature
        st.session_state.ocr_result = None

    image_path = save_uploaded_file(uploaded_file)
    st.session_state.ocr_image_path = image_path

    preview_col, output_col = st.columns([1.05, 1.2], gap="large")

    with preview_col:
        st.subheader("Input Preview")
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.caption(f"File size: {file_size_kb:.1f} KB")

        if show_debug:
            render_debug_panel(image_path)

    with output_col:
        st.subheader("OCR Output")
        st.info("The TrOCR model will load on the first run only. Later runs reuse the cached model.")

        if st.button("Run OCR", type="primary", use_container_width=True):
            with st.spinner("Loading model and extracting handwritten text..."):
                pipeline = get_pipeline(use_gpu=use_gpu)
                text = pipeline.run(
                    image_path,
                    "outputs/text/output.txt",
                    "outputs/pdf/output.pdf",
                )
                st.session_state.ocr_result = text

        if st.session_state.ocr_result is not None:
            st.success("OCR completed.")
            st.text_area(
                "Extracted Text",
                st.session_state.ocr_result,
                height=260,
            )

            with open("outputs/pdf/output.pdf", "rb") as pdf_file:
                st.download_button(
                    "Download PDF",
                    pdf_file,
                    file_name="output.pdf",
                    use_container_width=True,
                )

            st.download_button(
                "Download Text",
                st.session_state.ocr_result,
                file_name="output.txt",
                use_container_width=True,
            )
else:
    st.markdown(
        """
        <div class="hero-card">
            <h3 style="margin-top: 0; color: #14324a;">How this version is faster</h3>
            <p style="color: #35546d; margin-bottom: 0.4rem;">
                The TrOCR model is no longer created when the page first opens.
                It is loaded only after you click <strong>Run OCR</strong>, and then Streamlit keeps it cached.
            </p>
            <p style="color: #35546d; margin-bottom: 0;">
                Upload an image to preview it, inspect detected text lines, and run OCR when you are ready.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

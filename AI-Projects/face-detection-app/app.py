"""Streamlit app for stable real-time YOLO + CNN mask classification."""

from __future__ import annotations

import threading

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

from src.detector import FaceDetector
from src.mask_predictor import MaskPredictor

st.set_page_config(page_title="Mask Vision Monitor", page_icon="CV", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.15), transparent 30%),
            radial-gradient(circle at top right, rgba(2, 132, 199, 0.14), transparent 26%),
            linear-gradient(180deg, #f7fbfc 0%, #edf6f9 100%);
    }
    .hero-card, .panel-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
        padding: 1.2rem 1.3rem;
    }
    .hero-card { margin-bottom: 1rem; }
    .eyebrow {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0f766e;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .hero-subtitle {
        color: #334155;
        font-size: 1rem;
        margin-top: 0.45rem;
        margin-bottom: 0;
    }
    .status-chip {
        display: inline-block;
        padding: 0.34rem 0.72rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
    }
    .chip-ready {
        color: #166534;
        background: rgba(34, 197, 94, 0.14);
    }
    .chip-warn {
        color: #9a3412;
        background: rgba(249, 115, 22, 0.16);
    }
    .chip-info {
        color: #0f172a;
        background: rgba(148, 163, 184, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Production Pipeline</div>
        <h1 class="hero-title">YOLO -> Crop -> CNN Mask Vision Monitor</h1>
        <p class="hero-subtitle">
            Stable browser webcam streaming with YOLO detection and mask / no-mask classification on cropped regions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_yolo_model() -> YOLO:
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_mask_predictor() -> MaskPredictor:
    return MaskPredictor(model_path="models/mask_model.pth", device="cpu")


@st.cache_resource
def load_detector() -> FaceDetector:
    return FaceDetector(load_yolo_model(), load_mask_predictor(), conf_threshold=0.4)


detector = load_detector()
mask_predictor = load_mask_predictor()
frame_lock = threading.Lock()

left_col, right_col = st.columns([0.95, 2.05], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("System Status")
    st.caption("The webcam feed is handled in-browser through WebRTC, which is much more stable than rerun-driven OpenCV capture in Streamlit.")

    st.markdown('<span class="status-chip chip-ready">YOLO Loaded</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-chip chip-info">Inference on CPU</span>', unsafe_allow_html=True)
    if mask_predictor.available:
        st.markdown('<span class="status-chip chip-ready">Classifier Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-chip chip-warn">Classifier Missing</span>', unsafe_allow_html=True)
        st.warning("`models/mask_model.pth` was not found. Live detection will still run, but mask classification will be unavailable.")

    st.metric("Detector", "YOLOv8")
    st.metric("Classifier Input", "128 x 128 RGB")
    st.metric("Transport", "WebRTC")

    st.markdown(
        """
        **How to use**

        1. Click `START` under the live monitor.
        2. Allow browser camera permission.
        3. Watch the annotated live stream update in real time.
        4. Click `STOP` when you are done.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Live Monitor")
    st.caption("If the browser asks for camera permission, choose Allow.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        with frame_lock:
            processed = detector.detect_and_draw(image)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

    webrtc_streamer(
        key="mask-detection-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


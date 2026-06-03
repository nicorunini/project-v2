import os
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(page_title="Eye State Model Benchmark", layout="wide")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "cnn_classifier.h5"
DEFAULT_LABELS_PATH = "labels.txt"
FALLBACK_LABELS = ["Closed", "Open", "Partially Closed"]
IMAGE_SIZE = (128, 128)
MODEL_LOCK = threading.Lock()

BENCHMARK_MODELS = {
    "custom_cnn": {
        "label": "Custom CNN",
        "path": "trained_models/custom_cnn.h5",
        "description": "Baseline · 2 conv layers from scratch",
        "color": "#378ADD",
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0",
        "path": "trained_models/efficientnet_b0.h5",
        "description": "Transfer learning · Compound scaling",
        "color": "#1D9E75",
    },
    "mobilenet_v2": {
        "label": "MobileNetV2",
        "path": "trained_models/mobilenet_v2.h5",
        "description": "Transfer learning · Inverted residuals",
        "color": "#D85A30",
    },
    "densenet_121": {
        "label": "DenseNet-121",
        "path": "trained_models/densenet_121.h5",
        "description": "Transfer learning · Dense connections",
        "color": "#7F77DD",
    },
}


# ---------------------------------------------------------------------------
# Shared helpers (unchanged from original)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def parse_labels(raw_labels: str):
    labels = [label.strip() for label in raw_labels.splitlines() if label.strip()]
    return labels or FALLBACK_LABELS


def load_labels(labels_path: str):
    path = Path(labels_path)
    if path.exists():
        return parse_labels(path.read_text(encoding="utf-8"))
    return FALLBACK_LABELS


def suggested_labels_path(model_path: str):
    model_stem = Path(model_path).stem
    model_labels_path = Path(f"{model_stem}_labels.txt")
    if model_labels_path.exists():
        return str(model_labels_path)
    return DEFAULT_LABELS_PATH


def model_choices():
    search_dirs = [Path("."), Path.cwd().parent]
    choices = sorted({
        str(path)
        for search_dir in search_dirs
        for pattern in ("*.keras", "*.h5")
        for path in search_dir.glob(pattern)
    })
    if DEFAULT_MODEL_PATH in choices:
        choices.remove(DEFAULT_MODEL_PATH)
    choices.insert(0, DEFAULT_MODEL_PATH)
    return choices


def expected_channels(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    channels = input_shape[-1]
    return int(channels) if channels in (1, 3) else 3


def model_has_rescaling(model):
    return any(layer.__class__.__name__ == "Rescaling" for layer in model.layers)


def prepare_image(image: Image.Image, normalize_input: bool, channels: int):
    image_mode = "L" if channels == 1 else "RGB"
    image = image.convert(image_mode).resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    if channels == 1:
        array = np.expand_dims(array, axis=-1)
    if normalize_input:
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def as_probabilities(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 1:
        positive = float(np.clip(scores[0], 0.0, 1.0))
        return np.array([1.0 - positive, positive], dtype=np.float32)
    if np.any(scores < 0.0) or not np.isclose(float(np.sum(scores)), 1.0, atol=0.05):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / np.sum(exp_scores)
    return np.clip(scores, 0.0, 1.0)


def predict(model, image: Image.Image, normalize_input: bool):
    prepared = prepare_image(image, normalize_input, expected_channels(model))
    with MODEL_LOCK:
        prediction = model.predict(prepared, verbose=0)
    return as_probabilities(prediction)


def prediction_table(labels, scores):
    best_index = int(np.argmax(scores))
    best_label = labels[best_index]
    best_score = float(scores[best_index])

    st.subheader(f"Prediction: {best_label}")
    st.metric("Confidence", f"{best_score:.2%}")

    st.write("Scores")
    for label, score in zip(labels, scores):
        score = float(score)
        st.progress(score)
        st.write(f"{label}: {score:.2%}")


def draw_prediction(image: Image.Image, label: str, score: float):
    frame = image.convert("RGB")
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    text = f"{label}  {score:.1%}"
    padding = 8
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + (padding * 2)
    height = box[3] - box[1] + (padding * 2)
    draw.rectangle((8, 8, 8 + width, 8 + height), fill=(15, 23, 42))
    draw.text((8 + padding, 8 + padding), text, fill=(255, 255, 255), font=font)
    return frame


def run_live_camera(model, labels, normalize_input: bool):
    try:
        import av
        from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.error("Live webcam mode needs `streamlit-webrtc`.")
        st.code("pip install streamlit-webrtc", language="bash")
        return

    class EyeStateVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            image_array = frame.to_ndarray(format="rgb24")
            image = Image.fromarray(image_array)
            scores = predict(model, image, normalize_input)
            frame_labels = labels
            if len(frame_labels) != len(scores):
                frame_labels = [f"Class {index}" for index in range(len(scores))]
            best_index = int(np.argmax(scores))
            annotated = draw_prediction(image, frame_labels[best_index], float(scores[best_index]))
            return av.VideoFrame.from_ndarray(np.asarray(annotated), format="rgb24")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_streamer(
        key="eye-state-live-camera",
        video_processor_factory=EyeStateVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_configuration,
    )


# ---------------------------------------------------------------------------
# Benchmark tab helpers
# ---------------------------------------------------------------------------

def benchmark_model_card(key, info, result, labels):
    color = info["color"]
    pred_label = labels[int(np.argmax(result["scores"]))]
    confidence = float(np.max(result["scores"]))

    st.markdown(
        f"<div style='border:2px solid {color};border-radius:10px;"
        f"padding:14px 16px;'>"
        f"<div style='font-size:13px;font-weight:600;color:{color};margin-bottom:2px;'>"
        f"{info['label']}</div>"
        f"<div style='font-size:11px;color:gray;margin-bottom:10px;'>"
        f"{info['description']}</div>"
        f"<div style='font-size:24px;font-weight:700;color:{color};'>{pred_label}</div>"
        f"<div style='font-size:13px;color:gray;margin-bottom:6px;'>"
        f"{confidence:.1%} confidence</div>"
        f"<div style='font-size:11px;color:gray;'>⏱ {result['inference_ms']:.1f} ms</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    for label, score in zip(labels, result["scores"]):
        st.caption(f"{label}: {float(score):.1%}")
        st.progress(float(score))


def run_benchmark_tab(labels):
    st.subheader("Multi-model benchmark")
    st.caption(
        "Upload one eye image and run it through all four models simultaneously. "
        "Train the models first with `python train_model.py`."
    )

    bench_source = st.radio(
        "Image source",
        ["Upload image", "Camera snapshot"],
        horizontal=True,
        key="bench_source",
    )

    bench_image = None
    if bench_source == "Upload image":
        uploaded = st.file_uploader(
            "Choose a cropped eye image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="bench_uploader",
        )
        if uploaded:
            bench_image = Image.open(uploaded)
    else:
        camera_file = st.camera_input("Take a photo", key="bench_camera")
        if camera_file:
            bench_image = Image.open(camera_file)

    if bench_image is None:
        st.info("Upload or capture an eye image to start the benchmark.")
        return

    col_img, _ = st.columns([1, 3])
    with col_img:
        st.image(bench_image, caption="Input image", use_column_width=True)

    st.write("**Select models to run**")
    check_cols = st.columns(len(BENCHMARK_MODELS))
    selected_keys = []
    for col, (key, info) in zip(check_cols, BENCHMARK_MODELS.items()):
        with col:
            model_file_exists = Path(info["path"]).exists()
            checked = st.checkbox(
                info["label"],
                value=model_file_exists,
                disabled=not model_file_exists,
                key=f"bench_check_{key}",
            )
            if not model_file_exists:
                st.caption("❌ Not trained yet")
            if checked and model_file_exists:
                selected_keys.append(key)

    if not selected_keys:
        st.warning("No trained models found. Run `python train_model.py` first.")
        return

    if not st.button("Run benchmark", type="primary"):
        return

    results = {}
    progress = st.progress(0, text="Loading models...")

    for i, key in enumerate(selected_keys):
        info = BENCHMARK_MODELS[key]
        progress.progress(i / len(selected_keys), text=f"Running {info['label']}...")
        try:
            model = load_model(info["path"])
            normalize = not model_has_rescaling(model)
            t0 = time.time()
            scores = predict(model, bench_image, normalize)
            elapsed_ms = (time.time() - t0) * 1000
            results[key] = {"scores": scores, "inference_ms": elapsed_ms}
        except Exception as exc:
            st.error(f"Failed to run {info['label']}: {exc}")

    progress.progress(1.0, text="Done!")

    if not results:
        return

    st.divider()
    st.write("**Results**")
    result_cols = st.columns(len(results))
    for col, (key, result) in zip(result_cols, results.items()):
        with col:
            benchmark_model_card(key, BENCHMARK_MODELS[key], result, labels)

    # Comparison table
    st.divider()
    st.write("**Side-by-side comparison**")
    table = {"Model": [], "Prediction": [], "Confidence": []}
    for cls in labels:
        table[cls] = []
    table["Inference (ms)"] = []

    for key, result in results.items():
        pred_idx = int(np.argmax(result["scores"]))
        table["Model"].append(BENCHMARK_MODELS[key]["label"])
        table["Prediction"].append(labels[pred_idx])
        table["Confidence"].append(f"{float(result['scores'][pred_idx]):.1%}")
        for cls, score in zip(labels, result["scores"]):
            table[cls].append(f"{float(score):.1%}")
        table["Inference (ms)"].append(f"{result['inference_ms']:.1f}")

    st.dataframe(table, use_container_width=True)

    # Agreement check
    predictions = [labels[int(np.argmax(r["scores"]))] for r in results.values()]
    if len(set(predictions)) == 1:
        st.success(f"All models agree: **{predictions[0]}**")
    else:
        votes = {}
        for p in predictions:
            votes[p] = votes.get(p, 0) + 1
        majority = max(votes, key=votes.get)
        st.warning(
            f"Models disagree. Majority vote: **{majority}** ({votes[majority]}/{len(predictions)} models). "
            + " · ".join(
                f"{BENCHMARK_MODELS[k]['label']} → {labels[int(np.argmax(r['scores']))]}"
                for k, r in results.items()
            )
        )

    # Export
    st.divider()
    import json
    export = {
        "labels": labels,
        "results": {
            k: {
                "model": BENCHMARK_MODELS[k]["label"],
                "prediction": labels[int(np.argmax(r["scores"]))],
                "confidence": round(float(np.max(r["scores"])), 4),
                "probabilities": {
                    cls: round(float(s), 4)
                    for cls, s in zip(labels, r["scores"])
                },
                "inference_ms": round(r["inference_ms"], 2),
            }
            for k, r in results.items()
        },
    }
    st.download_button(
        "Download results as JSON",
        data=json.dumps(export, indent=2),
        file_name="benchmark_results.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("Eye State Model Benchmark")

bench_labels = load_labels(DEFAULT_LABELS_PATH)
run_benchmark_tab(bench_labels)

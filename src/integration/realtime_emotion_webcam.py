"""
Run real-time webcam emotion detection with a trained Keras model.

The script is intentionally conservative for local Windows inference:
- CPU inference is used by default to avoid TensorFlow CUDA DLL noise.
- .h5 models saved by newer Keras are sanitized before loading in TF 2.10.
- The requested --model path is loaded directly; the script does not silently
  fall back to a different model.
"""

import argparse
import json
import os
import platform
import shutil
import tempfile
import time
from collections import deque

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import cv2
    import h5py
    import numpy as np
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"Missing Python dependency: {exc.name}. Activate the project virtual "
        "environment and install requirements.txt."
    ) from exc


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "emotion_cnn_residual.h5")

CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

NEWER_KERAS_KEYS_TO_DROP = {
    "batch_shape",
    "optional",
    "synchronized",
    "quantization_config",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time emotion detection from webcam.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained .h5 or .keras emotion model.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index. Use 0 for the default webcam.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="OpenCV camera backend. On Windows, dshow is often more reliable than msmf.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Minimum detected face size in pixels.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Haar detector scale factor. Lower values are slower but can detect faces more accurately.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=6,
        help="Haar detector strictness. Lower values detect more faces but may add false positives.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.20,
        help="Face crop margin around the detected box.",
    )
    parser.add_argument(
        "--all-faces",
        action="store_true",
        help="Predict every detected face. By default only the largest face is used for a cleaner demo.",
    )
    parser.add_argument(
        "--equalize-hist",
        action="store_true",
        help="Apply histogram equalization before face detection. Useful only in difficult lighting.",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=8,
        help="Number of recent prediction vectors to average.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["auto", "rescale", "raw"],
        default="auto",
        help=(
            "rescale: divide pixels by 255 for custom CNNs. "
            "raw: pass 0-255 float32 pixels for EfficientNet models with preprocessing inside. "
            "auto: use raw for EfficientNet-looking model names, otherwise rescale."
        ),
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional folder for saving preprocessed face crops.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show the camera feed without running face detection or prediction.",
    )
    parser.add_argument(
        "--allow-dark-camera",
        action="store_true",
        help="Accept very dark camera frames instead of trying another backend/resolution.",
    )
    return parser.parse_args()


def choose_preprocess_mode(model_path, requested_mode):
    if requested_mode != "auto":
        return requested_mode

    filename = os.path.basename(model_path).lower()
    if "efficientnet" in filename:
        return "raw"
    return "rescale"


def load_emotion_model(model_path):
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Emotion model not found: {model_path}")

    print("TensorFlow version:", tf.__version__)
    print("Loading emotion model:", model_path)

    custom_objects = build_compatibility_objects()
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
        )
    except (TypeError, ValueError) as exc:
        if not model_path.lower().endswith(".h5"):
            raise RuntimeError(
                f"Could not load {model_path}. This usually means the model was saved "
                "with a newer Keras version than this environment supports. Try a .h5 "
                "export or use a newer inference virtual environment."
            ) from exc

        print("Direct H5 load failed. Trying sanitized H5 compatibility copy.")
        print("Original load error:", exc)
        sanitized_path = create_sanitized_h5_copy(model_path)
        try:
            return tf.keras.models.load_model(
                sanitized_path,
                compile=False,
                custom_objects=custom_objects,
            )
        except Exception as second_exc:
            raise RuntimeError(
                "Could not load the H5 model even after sanitizing newer Keras "
                "serialization fields. Use a Windows inference environment with a "
                "TensorFlow/Keras version closer to the Ubuntu training environment, "
                "or re-export the model from Ubuntu using the same TensorFlow version "
                "as this Windows environment."
            ) from second_exc


def build_compatibility_objects():
    class CompatibleInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, batch_shape=None, optional=None, **kwargs):
            if batch_shape is not None and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = batch_shape
            super().__init__(*args, **kwargs)

    class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
        def __init__(self, *args, synchronized=None, **kwargs):
            super().__init__(*args, **kwargs)

    return {
        "InputLayer": CompatibleInputLayer,
        "BatchNormalization": CompatibleBatchNormalization,
        "DTypePolicy": tf.keras.mixed_precision.Policy,
    }


def create_sanitized_h5_copy(model_path):
    temp_dir = tempfile.mkdtemp(prefix="emotion_model_")
    sanitized_path = os.path.join(temp_dir, os.path.basename(model_path))
    shutil.copy2(model_path, sanitized_path)

    with h5py.File(sanitized_path, "r+") as h5_file:
        if "model_config" not in h5_file.attrs:
            raise ValueError("H5 file does not contain a Keras model_config attribute.")

        raw_config = h5_file.attrs["model_config"]
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")

        model_config = json.loads(raw_config)
        sanitize_keras_config(model_config)
        del h5_file.attrs["model_config"]
        h5_file.attrs["model_config"] = json.dumps(model_config)

    return sanitized_path


def sanitize_keras_config(value):
    if isinstance(value, dict):
        config = value.get("config")
        if isinstance(config, dict):
            if "batch_shape" in config and "batch_input_shape" not in config:
                config["batch_input_shape"] = config["batch_shape"]
            for key in NEWER_KERAS_KEYS_TO_DROP:
                config.pop(key, None)

        for key in list(value.keys()):
            if key in NEWER_KERAS_KEYS_TO_DROP:
                value.pop(key, None)
            else:
                sanitize_keras_config(value[key])

    elif isinstance(value, list):
        for item in value:
            sanitize_keras_config(item)


def get_model_input_size(model):
    input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]
    if height is None or width is None or channels != 3:
        raise ValueError(f"Unsupported model input shape: {input_shape}")
    print(f"Model input size: {height}x{width}x{channels}")
    return int(width), int(height)


def build_face_detector():
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load OpenCV face detector: {cascade_path}")
    return detector


def camera_backend_candidates(backend_name):
    if backend_name == "dshow":
        return [("DirectShow", cv2.CAP_DSHOW)]
    if backend_name == "msmf":
        return [("MSMF", cv2.CAP_MSMF)]
    if backend_name == "any":
        return [("Default", cv2.CAP_ANY)]

    if platform.system().lower() == "windows":
        return [
            ("DirectShow", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("Default", cv2.CAP_ANY),
        ]
    return [("Default", cv2.CAP_ANY)]


def open_camera(camera_index, backend_name, allow_dark_camera=False):
    for backend_label, backend in camera_backend_candidates(backend_name):
        for width, height in ((640, 480), (1280, 720), (320, 240)):
            print(f"Trying camera {camera_index} with {backend_label} backend at {width}x{height}...")
            cap = cv2.VideoCapture(camera_index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

            best_brightness = -1.0
            best_frame_ok = False
            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    best_frame_ok = True
                    best_brightness = max(best_brightness, float(frame.mean()))
                    if allow_dark_camera or best_brightness > 5.0:
                        print(
                            f"Opened webcam using {backend_label} backend at "
                            f"{width}x{height}. Brightness mean: {best_brightness:.2f}"
                        )
                        return cap
                time.sleep(0.05)

            if best_frame_ok:
                print(
                    f"Camera opened with {backend_label} at {width}x{height}, "
                    f"but frames were almost black. Brightness mean: {best_brightness:.2f}"
                )

            cap.release()

    raise RuntimeError(
        f"Could not open webcam with index {camera_index}. "
        "Close other apps using the camera, check Windows camera privacy settings, "
        "check the physical privacy shutter, improve lighting, or try --backend dshow / --backend msmf."
    )


def get_square_face_box(frame, x, y, w, h, margin):
    frame_h, frame_w = frame.shape[:2]
    side = int(max(w, h) * (1.0 + 2.0 * margin))
    center_x = x + w // 2
    center_y = y + h // 2

    x1 = max(center_x - side // 2, 0)
    y1 = max(center_y - side // 2, 0)
    x2 = min(x1 + side, frame_w)
    y2 = min(y1 + side, frame_h)
    x1 = max(x2 - side, 0)
    y1 = max(y2 - side, 0)

    return x1, y1, x2, y2


def crop_face_square(frame, x, y, w, h, margin):
    x1, y1, x2, y2 = get_square_face_box(frame, x, y, w, h, margin)
    return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def preprocess_face(face_bgr, img_size, preprocess_mode):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, img_size, interpolation=cv2.INTER_AREA)
    face_rgb = face_rgb.astype(np.float32)

    if preprocess_mode == "rescale":
        face_rgb = face_rgb / 255.0

    return np.expand_dims(face_rgb, axis=0)


class PredictionSmoother:
    def __init__(self, window_size):
        self.history = deque(maxlen=max(1, int(window_size)))

    def update(self, probabilities):
        self.history.append(probabilities)
        return np.mean(np.array(self.history), axis=0)


def predict_emotion(model, face_bgr, img_size, preprocess_mode, smoother):
    batch = preprocess_face(face_bgr, img_size, preprocess_mode)
    probabilities = model.predict(batch, verbose=0)[0]
    probabilities = smoother.update(probabilities)

    top_indices = np.argsort(probabilities)[::-1]
    top_predictions = [
        (CLASSES[index], float(probabilities[index]))
        for index in top_indices
        if index < len(CLASSES)
    ]
    return top_predictions


def select_faces(faces, frame_shape, all_faces=False):
    if len(faces) == 0:
        return []

    frame_h, frame_w = frame_shape[:2]
    max_box_area = frame_w * frame_h * 0.45
    min_aspect = 0.75
    max_aspect = 1.35
    filtered_faces = []

    for x, y, w, h in faces:
        aspect_ratio = w / float(h)
        box_area = w * h
        if box_area > max_box_area:
            continue
        if not (min_aspect <= aspect_ratio <= max_aspect):
            continue
        filtered_faces.append((x, y, w, h))

    faces = filtered_faces if filtered_faces else list(faces)
    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    if all_faces:
        return faces
    return faces[:1]


def draw_prediction(frame, x, y, w, h, top_predictions):
    color = (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if not top_predictions:
        return

    text_y = max(y - 10, 24)
    emotion, confidence = top_predictions[0]
    label = f"{emotion}: {confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def save_debug_crop(debug_dir, face_bgr, img_size, preprocess_mode, frame_count):
    os.makedirs(debug_dir, exist_ok=True)
    crop = preprocess_face(face_bgr, img_size, preprocess_mode)[0]
    if preprocess_mode == "rescale":
        crop = crop * 255.0
    crop_bgr = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(debug_dir, f"face_crop_{frame_count:05d}.jpg"), crop_bgr)


def run_webcam(model, args, img_size, preprocess_mode):
    detector = build_face_detector()
    cap = open_camera(args.camera, args.backend, args.allow_dark_camera)
    smoother = PredictionSmoother(args.smoothing)
    frame_count = 0

    print("Webcam started. Press 'q' to quit.")
    print("Preprocess mode:", preprocess_mode)
    if preprocess_mode == "raw":
        print("Passing 0-255 float32 RGB pixels into the model.")
    else:
        print("Passing 0-1 rescaled RGB pixels into the model.")
    print(f"Face crops are resized to {img_size[0]}x{img_size[1]} RGB before prediction.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from webcam. Retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        brightness = float(frame.mean())
        if frame_count % 60 == 1:
            print(f"Frame brightness mean: {brightness:.2f}")

        if args.preview_only:
            if brightness <= 5.0:
                cv2.putText(
                    frame,
                    "Camera frames are black - check shutter/privacy/lighting",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                "Preview only - press q to quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Real-time Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.equalize_hist:
            gray = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(args.min_face_size, args.min_face_size),
        )

        selected_faces = select_faces(faces, frame.shape, all_faces=args.all_faces)
        for x, y, w, h in selected_faces:
            face, _ = crop_face_square(frame, x, y, w, h, args.margin)
            if face.size == 0:
                continue

            top_predictions = predict_emotion(model, face, img_size, preprocess_mode, smoother)
            draw_prediction(frame, x, y, w, h, top_predictions)

            if args.debug_dir and frame_count % 30 == 0:
                save_debug_crop(args.debug_dir, face, img_size, preprocess_mode, frame_count)

        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Real-time Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    preprocess_mode = choose_preprocess_mode(args.model, args.preprocess)
    if args.preview_only:
        model = None
        img_size = (96, 96)
    else:
        model = load_emotion_model(args.model)
        img_size = get_model_input_size(model)
    run_webcam(model, args, img_size, preprocess_mode)


if __name__ == "__main__":
    main()

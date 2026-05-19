"""
Reusable OOP emotion detection module for the attendance system.

This module is designed for final integration with the group attendance pipeline.

Main final-use method:
    predict_from_face(face_bgr)

Temporary testing method:
    predict_from_frame(frame_bgr)

It does not open the webcam by itself.
"""

import json
import os
import shutil
import tempfile
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


# ----------------------------------------------------------------------
# Paths and model configuration
# ----------------------------------------------------------------------
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


EMOTION_FEEDBACK = {
    "happy": {
        "emoji": "😊",
        "message": "You look happy today!",
    },
    "neutral": {
        "emoji": "😐",
        "message": "Neutral mood detected.",
    },
    "sad": {
        "emoji": "😟",
        "message": "You seem a bit low. Take a short break.",
    },
    "anger": {
        "emoji": "😠",
        "message": "You seem frustrated. Try taking a deep breath.",
    },
    "fear": {
        "emoji": "😨",
        "message": "Anxious expression detected.",
    },
    "surprise": {
        "emoji": "😲",
        "message": "Something caught your attention!",
    },
    "disgust": {
        "emoji": "🤢",
        "message": "Discomfort detected.",
    },
    "contempt": {
        "emoji": "🙄",
        "message": "Contempt-like expression detected.",
    },
}


NEWER_KERAS_KEYS_TO_DROP = {
    "batch_shape",
    "optional",
    "synchronized",
    "quantization_config",
}


# ----------------------------------------------------------------------
# Prediction smoother
# ----------------------------------------------------------------------
class PredictionSmoother:
    """
    Averages the last few prediction probability vectors.
    This reduces frame-by-frame flickering.
    """

    def __init__(self, window_size=3):
        self.history = deque(maxlen=max(1, int(window_size)))

    def update(self, probabilities):
        self.history.append(probabilities)
        return np.mean(np.array(self.history), axis=0)

    def reset(self):
        self.history.clear()


# ----------------------------------------------------------------------
# Stable emotion tracker
# ----------------------------------------------------------------------
class StableEmotionTracker:
    """
    Stabilizes emotion output for attendance-style use.

    Behaviour:
    - Does not display 'uncertain'.
    - If the prediction is weak or confused, it keeps the previous stable emotion.
    - It only changes emotion after the same confident emotion appears for several frames.
    """

    def __init__(self, required_frames=5, min_confidence=0.40, min_margin=0.08):
        self.required_frames = required_frames
        self.min_confidence = min_confidence
        self.min_margin = min_margin

        self.current_emotion = "neutral"
        self.current_confidence = 0.0

        self.candidate_emotion = None
        self.candidate_confidence = 0.0
        self.candidate_count = 0

    def update(self, top_predictions):
        if not top_predictions:
            return self.current_emotion, self.current_confidence

        top_emotion, top_confidence = top_predictions[0]

        second_confidence = 0.0
        if len(top_predictions) > 1:
            second_confidence = top_predictions[1][1]

        confidence_gap = top_confidence - second_confidence

        # If model is unsure, keep the last stable emotion.
        if top_confidence < self.min_confidence or confidence_gap < self.min_margin:
            return self.current_emotion, self.current_confidence

        if top_emotion == self.candidate_emotion:
            self.candidate_count += 1
            self.candidate_confidence = top_confidence
        else:
            self.candidate_emotion = top_emotion
            self.candidate_confidence = top_confidence
            self.candidate_count = 1

        if self.candidate_count >= self.required_frames:
            self.current_emotion = self.candidate_emotion
            self.current_confidence = self.candidate_confidence

        return self.current_emotion, self.current_confidence

    def reset(self):
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.candidate_emotion = None
        self.candidate_confidence = 0.0
        self.candidate_count = 0


# ----------------------------------------------------------------------
# Main reusable emotion detector
# ----------------------------------------------------------------------
class EmotionDetector:
    """
    Reusable OOP emotion detector.

    Final team integration:
        detector = EmotionDetector()
        result = detector.predict_from_face(face_crop)

    Temporary standalone testing:
        result = detector.predict_from_frame(frame)
    """

    def __init__(
        self,
        model_path=None,
        preprocess_mode="rescale",
        smoothing=3,
        stable_frames=5,
        min_confidence=0.40,
        min_margin=0.08,
        face_margin=0.20,
        min_face_size=80,
    ):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model_path = os.path.abspath(model_path)
        self.preprocess_mode = preprocess_mode
        self.face_margin = face_margin
        self.min_face_size = min_face_size

        self.model = self._load_emotion_model(self.model_path)
        self.img_size = self._get_model_input_size(self.model)

        self.smoother = PredictionSmoother(window_size=smoothing)
        self.tracker = StableEmotionTracker(
            required_frames=stable_frames,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )

        self.face_detector = self._build_face_detector()

    # ------------------------------------------------------------------
    # Final integration method
    # ------------------------------------------------------------------
    def predict_from_face(self, face_bgr):
        """
        Predict emotion from a cropped face.

        Input:
            face_bgr:
                Cropped OpenCV face image in BGR format.

        Output:
            Dictionary with:
                face_detected
                emotion
                confidence
                emoji
                message
                face_box
                top_predictions
        """

        if face_bgr is None or face_bgr.size == 0:
            return self._empty_result(face_detected=False)

        batch = self._preprocess_face(face_bgr)

        probabilities = self.model.predict(batch, verbose=0)[0]
        probabilities = self.smoother.update(probabilities)

        top_predictions = self._get_top_predictions(probabilities)
        stable_emotion, stable_confidence = self.tracker.update(top_predictions)

        return self._format_result(
            emotion=stable_emotion,
            confidence=stable_confidence,
            face_detected=True,
            face_box=None,
            top_predictions=top_predictions,
        )

    # ------------------------------------------------------------------
    # Temporary testing method
    # ------------------------------------------------------------------
    def predict_from_frame(self, frame_bgr):
        """
        Predict emotion from a full webcam frame.

        This is useful while testing alone, before teammates provide a cropped face.
        It detects the largest face, crops it, and then calls predict_from_face().
        """

        if frame_bgr is None or frame_bgr.size == 0:
            return self._empty_result(face_detected=False)

        face_data = self._detect_largest_face(frame_bgr)

        if face_data is None:
            self.reset()
            return self._empty_result(face_detected=False)

        face_crop, face_box = face_data

        result = self.predict_from_face(face_crop)
        result["face_box"] = face_box

        return result

    def reset(self):
        """
        Reset smoother and stable emotion tracker.

        Call this when:
        - no face is detected,
        - a different person enters the frame,
        - attendance has just been marked and you want a fresh state.
        """

        self.smoother.reset()
        self.tracker.reset()

    # ------------------------------------------------------------------
    # Face detection helpers for predict_from_frame()
    # ------------------------------------------------------------------
    def _build_face_detector(self):
        cascade_path = os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml",
        )

        detector = cv2.CascadeClassifier(cascade_path)

        if detector.empty():
            raise RuntimeError(f"Failed to load OpenCV face detector: {cascade_path}")

        return detector

    def _detect_largest_face(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(self.min_face_size, self.min_face_size),
        )

        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        x, y, w, h = faces[0]

        face_crop, square_box = self._crop_face_square(frame_bgr, x, y, w, h)

        if face_crop is None or face_crop.size == 0:
            return None

        return face_crop, square_box

    def _crop_face_square(self, frame_bgr, x, y, w, h):
        frame_h, frame_w = frame_bgr.shape[:2]

        side = int(max(w, h) * (1.0 + 2.0 * self.face_margin))
        center_x = x + w // 2
        center_y = y + h // 2

        x1 = max(center_x - side // 2, 0)
        y1 = max(center_y - side // 2, 0)

        x2 = min(x1 + side, frame_w)
        y2 = min(y1 + side, frame_h)

        x1 = max(x2 - side, 0)
        y1 = max(y2 - side, 0)

        face_crop = frame_bgr[y1:y2, x1:x2]
        square_box = (x1, y1, x2 - x1, y2 - y1)

        return face_crop, square_box

    # ------------------------------------------------------------------
    # Preprocessing and prediction helpers
    # ------------------------------------------------------------------
    def _preprocess_face(self, face_bgr):
        """
        Convert OpenCV BGR face crop into model input.

        Handles:
        - BGR to RGB
        - resize to model input size
        - float32 conversion
        - /255 rescaling
        - batch dimension
        """

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, self.img_size, interpolation=cv2.INTER_AREA)
        face_rgb = face_rgb.astype(np.float32)

        if self.preprocess_mode == "rescale":
            face_rgb = face_rgb / 255.0

        return np.expand_dims(face_rgb, axis=0)

    def _get_top_predictions(self, probabilities):
        top_indices = np.argsort(probabilities)[::-1]

        return [
            (CLASSES[index], float(probabilities[index]))
            for index in top_indices
            if index < len(CLASSES)
        ]

    def _format_result(
        self,
        emotion,
        confidence,
        face_detected=True,
        face_box=None,
        top_predictions=None,
    ):
        info = EMOTION_FEEDBACK.get(emotion, EMOTION_FEEDBACK["neutral"])

        return {
            "face_detected": face_detected,
            "emotion": emotion,
            "confidence": round(float(confidence), 2),
            "emoji": info["emoji"],
            "message": info["message"],
            "face_box": face_box,
            "top_predictions": top_predictions if top_predictions is not None else [],
        }

    def _empty_result(self, face_detected=False):
        info = EMOTION_FEEDBACK["neutral"]

        return {
            "face_detected": face_detected,
            "emotion": "neutral",
            "confidence": 0.0,
            "emoji": info["emoji"],
            "message": info["message"],
            "face_box": None,
            "top_predictions": [],
        }

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _get_model_input_size(self, model):
        input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape

        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        if height is None or width is None or channels != 3:
            raise ValueError(f"Unsupported model input shape: {input_shape}")

        print(f"Emotion model input size: {height}x{width}x{channels}")

        return int(width), int(height)

    def _load_emotion_model(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Emotion model not found: {model_path}")

        print("TensorFlow version:", tf.__version__)
        print("Loading emotion model:", model_path)

        custom_objects = self._build_compatibility_objects()

        try:
            return tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
            )

        except (TypeError, ValueError) as exc:
            if not model_path.lower().endswith(".h5"):
                raise RuntimeError(
                    f"Could not load {model_path}. "
                    "Try using a compatible .h5 model export."
                ) from exc

            print("Direct H5 load failed. Trying sanitized H5 compatibility copy.")
            print("Original load error:", exc)

            sanitized_path = self._create_sanitized_h5_copy(model_path)

            try:
                return tf.keras.models.load_model(
                    sanitized_path,
                    compile=False,
                    custom_objects=custom_objects,
                )
            except Exception as second_exc:
                raise RuntimeError(
                    "Could not load the H5 model even after sanitizing newer Keras "
                    "serialization fields. Use a compatible TensorFlow/Keras "
                    "environment or re-export the model."
                ) from second_exc

    def _build_compatibility_objects(self):
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

    def _create_sanitized_h5_copy(self, model_path):
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
            self._sanitize_keras_config(model_config)

            del h5_file.attrs["model_config"]
            h5_file.attrs["model_config"] = json.dumps(model_config)

        return sanitized_path

    def _sanitize_keras_config(self, value):
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
                    self._sanitize_keras_config(value[key])

        elif isinstance(value, list):
            for item in value:
                self._sanitize_keras_config(item)
"""
Integration module for the EfficientNet-B0 emotion detector (Model 3).

Provides the same interface as emotion_detector.py so callers can swap
models without changing their code.

Usage:
    from src.integration.emotion_detector_efficientnet import EmotionDetectorEfficientNet
    detector = EmotionDetectorEfficientNet()
    result = detector.predict_from_face(face_bgr)
"""

import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_FEEDBACK = {
    "happy":    {"emoji": "😊", "message": "You look happy today!"},
    "neutral":  {"emoji": "😐", "message": "Neutral mood detected."},
    "sad":      {"emoji": "😟", "message": "You seem a bit low..."},
    "anger":    {"emoji": "😠", "message": "You seem frustrated..."},
    "fear":     {"emoji": "😨", "message": "Anxious expression detected."},
    "surprise": {"emoji": "😲", "message": "Something caught your attention!"},
    "disgust":  {"emoji": "🤢", "message": "Discomfort detected."},
    "contempt": {"emoji": "🙄", "message": "Contempt-like expression detected."},
}

DEFAULT_MODEL_PATHS = [
    "models/emotion_detection_2/efficientnet/emotion_efficientnet.keras",
    "models/emotion_detection_2/efficientnet/emotion_efficientnet.h5",
]


# ---------------------------------------------------------------------------
# Prediction smoothing / stability helpers
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """Moving-average smoother over a sliding window of raw probability vectors."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self._history.append(probs.copy())
        return np.mean(self._history, axis=0)

    def reset(self):
        self._history.clear()


class StableEmotionTracker:
    """Only changes the displayed emotion after N consecutive matching frames."""

    def __init__(self, stable_frames: int = 5, min_confidence: float = 0.40, min_margin: float = 0.08):
        self.stable_frames = stable_frames
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self._current_emotion: str | None = None
        self._pending_emotion: str | None = None
        self._pending_count: int = 0

    def update(self, emotion: str, confidence: float, margin: float) -> str | None:
        if confidence < self.min_confidence or margin < self.min_margin:
            return self._current_emotion

        if emotion == self._pending_emotion:
            self._pending_count += 1
        else:
            self._pending_emotion = emotion
            self._pending_count = 1

        if self._pending_count >= self.stable_frames:
            self._current_emotion = emotion
            self._pending_count = 0

        return self._current_emotion

    def reset(self):
        self._current_emotion = None
        self._pending_emotion = None
        self._pending_count = 0


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class EmotionDetectorEfficientNet:
    """EfficientNet-B0 emotion detector with temporal smoothing.

    Accepts a pre-cropped face image in BGR format (from OpenCV) and returns
    a prediction dict matching the interface of EmotionDetector in
    emotion_detector.py.
    """

    def __init__(
        self,
        model_path: str | None = None,
        smoothing: int = 3,
        stable_frames: int = 5,
        min_confidence: float = 0.40,
        min_margin: float = 0.08,
    ):
        self.model_path = model_path
        self.smoother = PredictionSmoother(window_size=smoothing)
        self.tracker = StableEmotionTracker(
            stable_frames=stable_frames,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        self.model = self._load_model()
        self.input_size = self._get_input_size()

    def _load_model(self) -> tf.keras.Model:
        candidates = [self.model_path] if self.model_path else DEFAULT_MODEL_PATHS
        for path in candidates:
            if path and os.path.exists(path):
                print(f"[EmotionDetectorEfficientNet] Loading model: {path}")
                return tf.keras.models.load_model(path, compile=False)
        raise FileNotFoundError(
            f"EfficientNet emotion model not found. Checked: {candidates}\n"
            "Train first: python src/training/emotion_detection_2/train_emotion_efficientnet.py"
        )

    def _get_input_size(self) -> tuple[int, int]:
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        _, h, w, _ = shape
        return int(w), int(h)

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, self.input_size, interpolation=cv2.INTER_AREA)
        face_float = face_resized.astype(np.float32) / 255.0
        return np.expand_dims(face_float, axis=0)

    def _top_predictions(self, probs: np.ndarray, top_k: int = 3):
        indices = np.argsort(probs)[::-1][:top_k]
        return [(CLASSES[i], float(probs[i])) for i in indices]

    def predict_from_face(self, face_bgr: np.ndarray) -> dict:
        """Run inference on a pre-cropped face image (BGR, uint8).

        Returns a dict with keys: emotion, confidence, emoji, message,
        top_predictions, face_detected.
        """
        if face_bgr is None or face_bgr.size == 0:
            return {
                "face_detected": False,
                "emotion": None,
                "confidence": 0.0,
                "emoji": "",
                "message": "No face provided.",
                "face_box": None,
                "top_predictions": [],
            }

        batch = self._preprocess(face_bgr)
        raw_probs = self.model.predict(batch, verbose=0)[0]
        smooth_probs = self.smoother.update(raw_probs)

        top = self._top_predictions(smooth_probs)
        top_emotion, top_conf = top[0]
        margin = top_conf - top[1][1] if len(top) > 1 else 1.0

        stable = self.tracker.update(top_emotion, top_conf, margin)
        display_emotion = stable or top_emotion
        feedback = EMOTION_FEEDBACK.get(display_emotion, {"emoji": "", "message": ""})

        return {
            "face_detected": True,
            "emotion": display_emotion,
            "confidence": float(smooth_probs[CLASSES.index(display_emotion)]),
            "emoji": feedback["emoji"],
            "message": feedback["message"],
            "face_box": None,
            "top_predictions": top,
        }

    def predict_from_frame(self, frame_bgr: np.ndarray) -> dict:
        """Detect the largest face in a frame and predict emotion.

        This is a convenience method for stand-alone testing.  Production
        pipelines should crop the face externally and call predict_from_face().
        """
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            self.reset()
            return {
                "face_detected": False,
                "emotion": None,
                "confidence": 0.0,
                "emoji": "",
                "message": "No face detected.",
                "face_box": None,
                "top_predictions": [],
            }

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame_bgr[y: y + h, x: x + w]
        result = self.predict_from_face(face_crop)
        result["face_box"] = (x, y, w, h)
        return result

    def reset(self):
        self.smoother.reset()
        self.tracker.reset()

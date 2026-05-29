"""
Integration module for the Hybrid CNN-Transformer emotion detector (Model 4).

Custom layers (PatchPositionEmbedding, TransformerBlock) are registered here
so tf.keras.models.load_model can deserialise the saved model correctly.

Provides the same interface as emotion_detector.py so callers can swap
models without changing their code.

Usage:
    from src.integration.emotion_detector_transformer import EmotionDetectorTransformer
    detector = EmotionDetectorTransformer()
    result = detector.predict_from_face(face_bgr)
"""

import math
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_FEEDBACK = {
    "happy":    {"emoji": "", "message": "You look happy today!"},
    "neutral":  {"emoji": "", "message": "Neutral mood detected."},
    "sad":      {"emoji": "", "message": "You seem a bit low..."},
    "anger":    {"emoji": "", "message": "You seem frustrated..."},
    "fear":     {"emoji": "", "message": "Anxious expression detected."},
    "surprise": {"emoji": "", "message": "Something caught your attention!"},
    "disgust":  {"emoji": "", "message": "Discomfort detected."},
    "contempt": {"emoji": "", "message": "Contempt-like expression detected."},
}

DEFAULT_MODEL_PATHS = [
    "models/emotion_detection_2/hybrid_transformer/emotion_hybrid_transformer.h5",
    "models/emotion_detection_2/hybrid_transformer/emotion_hybrid_transformer.keras",
]


# Custom layers

class PatchPositionEmbedding(layers.Layer):
    def __init__(self, num_patches: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_embed = layers.Embedding(num_patches, embed_dim)

    def build(self, input_shape):
        self.pos_embed.build((None,))
        super().build(input_shape)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_embed(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return cfg


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        self.drop1.build(input_shape)
        self.drop2.build(input_shape)
        # Keras 3: MultiHeadAttention.build takes (query_shape, value_shape) separately
        self.attention.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        normed = self.norm1(x)
        attn = self.attention(normed, normed, training=training)
        attn = self.drop1(attn, training=training)
        x = x + attn
        normed = self.norm2(x)
        ff = self.ffn(normed)
        ff = self.drop2(ff, training=training)
        return x + ff

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return cfg


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-7):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        progress = tf.maximum(0.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


CUSTOM_OBJECTS = {
    "PatchPositionEmbedding": PatchPositionEmbedding,
    "TransformerBlock": TransformerBlock,
    "WarmupCosineDecay": WarmupCosineDecay,
}


# Prediction smoothing / stability helpers

class PredictionSmoother:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self._history.append(probs.copy())
        return np.mean(self._history, axis=0)

    def reset(self):
        self._history.clear()


class StableEmotionTracker:
    def __init__(self, stable_frames: int = 3, min_confidence: float = 0.20, min_margin: float = 0.05):
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


# Main detector

class EmotionDetectorTransformer:
    """Hybrid CNN-Transformer emotion detector with temporal smoothing.

    Loads the trained model with custom layer registration and provides
    the same predict_from_face / predict_from_frame interface as
    EmotionDetector in emotion_detector.py.
    """

    def __init__(
        self,
        model_path: str | None = None,
        smoothing: int = 3,
        stable_frames: int = 3,
        min_confidence: float = 0.20,
        min_margin: float = 0.05,
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
        self.smoothing_enabled = True

    def _load_model(self) -> tf.keras.Model:
        if self.model_path:
            base = self.model_path.rsplit(".", 1)[0]
            candidates = [base + ".h5", base + ".keras"]
        else:
            candidates = DEFAULT_MODEL_PATHS
        for path in candidates:
            if path and os.path.exists(path):
                print(f"[EmotionDetectorTransformer] Loading model: {path}")
                return tf.keras.models.load_model(
                    path,
                    custom_objects=CUSTOM_OBJECTS,
                    compile=False,
                )
        raise FileNotFoundError(
            f"Hybrid Transformer emotion model not found. Checked: {candidates}\n"
            "Train first: python src/training/emotion_detection_2/train_emotion_hybrid_transformer.py"
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

        # Guard against NaN/Inf from corrupted model weights
        if not np.all(np.isfinite(raw_probs)):
            raw_probs = np.ones(len(CLASSES), dtype=np.float32) / len(CLASSES)

        if self.smoothing_enabled:
            probs = self.smoother.update(raw_probs)
            top = self._top_predictions(probs)
            top_emotion, top_conf = top[0]
            margin = top_conf - top[1][1] if len(top) > 1 else 1.0
            stable = self.tracker.update(top_emotion, top_conf, margin)
            display_emotion = stable or top_emotion
        else:
            probs = raw_probs
            top = self._top_predictions(probs)
            top_emotion, top_conf = top[0]
            display_emotion = top_emotion

        feedback = EMOTION_FEEDBACK.get(display_emotion, {"emoji": "", "message": ""})

        return {
            "face_detected": True,
            "emotion": display_emotion,
            "confidence": float(probs[CLASSES.index(display_emotion)]),
            "emoji": feedback["emoji"],
            "message": feedback["message"],
            "face_box": None,
            "top_predictions": top,
        }

    def predict_from_frame(self, frame_bgr: np.ndarray) -> dict:
        """Detect the largest face in a frame and predict emotion."""
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

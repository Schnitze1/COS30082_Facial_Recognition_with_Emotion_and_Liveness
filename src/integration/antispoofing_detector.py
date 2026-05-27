"""
Anti-Spoofing (Liveness) Detector — MobileNetV2 binary classifier.

Predicts whether a face crop is a real person or a spoof (photo/screen/mask).

Usage:
    from src.integration.antispoofing_detector import AntispoofingDetector
    detector = AntispoofingDetector()
    result = detector.predict(face_bgr)
    # {"label": "real", "score": 0.92, "is_real": True}
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATHS = [
    BASE_DIR / "models" / "anti-spoofing" / "antispoofing_model.h5",
    BASE_DIR / "models" / "anti-spoofing" / "antispoofing_model.keras",
]

IMG_SIZE = 128
THRESHOLD = 0.1


class AntispoofingDetector:
    """MobileNetV2-based liveness detector.

    Accepts a BGR face crop (from OpenCV) and returns a dict with:
        label   : "real" | "spoof"
        score   : float — probability of being real (0–1)
        is_real : bool
        decision: "allow" | "block"  (matches GlassesDetector interface)
        message : str
    """

    def __init__(self, model_path=None, threshold: float = THRESHOLD):
        if model_path:
            base = str(model_path).rsplit(".", 1)[0]
            candidates = [Path(base + ".h5"), Path(base + ".keras")]
        else:
            candidates = DEFAULT_MODEL_PATHS

        loaded = None
        for p in candidates:
            if p.exists():
                print(f"[AntispoofingDetector] Loading model: {p}")
                loaded = tf.keras.models.load_model(str(p), compile=False)
                break

        if loaded is None:
            raise FileNotFoundError(
                f"Anti-spoofing model not found. Checked: {candidates}\n"
                "Train first: python src/training/anti_spoofing/train_antispoofing.py"
            )

        self.model = loaded
        self.threshold = threshold

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        face_rgb = face_rgb.astype(np.float32) / 255.0
        return np.expand_dims(face_rgb, axis=0)

    def predict(self, face_bgr: np.ndarray) -> dict:
        """Run liveness detection on a BGR face crop."""
        batch = self._preprocess(face_bgr)
        score = float(self.model.predict(batch, verbose=0)[0][0])
        is_real = score >= self.threshold
        label = "real" if is_real else "spoof"
        return {
            "label": label,
            "score": round(score, 4),
            "is_real": is_real,
            "decision": "allow" if is_real else "block",
            "message": "Liveness check passed." if is_real else "Spoof detected — please use a real face.",
        }

    def predict_with_decision(self, face_bgr: np.ndarray) -> dict:
        """Alias matching the GlassesDetector interface used by hybrid_attendance."""
        return self.predict(face_bgr)

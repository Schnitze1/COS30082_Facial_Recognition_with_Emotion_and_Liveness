"""
Anti-Spoofing Detector
Loads the trained MobileNetV2 model and provides a simple interface
for teammates to use in the GUI/integration layer.

Usage:
    from src.integration.antispoofing_detector import AntispoofingDetector

    detector = AntispoofingDetector()
    result = detector.predict(face_image)
    print(result)  # {"label": "real", "score": 0.92, "is_real": True}
"""

import sys
import logging
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]   # project root
MODEL_PATH = BASE_DIR / "models" / "anti_spoofing" / "antispoofing_model.keras"

# Fall back to .h5 if .keras not found
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "models" / "anti_spoofing" / "antispoofing_model.h5"

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE  = 128       # must match training
THRESHOLD = 0.1      # score >= threshold → real, below → spoof


class AntispoofingDetector:
    """
    Liveness detector for the attendance system.

    Example
    -------
    detector = AntispoofingDetector()

    # Pass a BGR face crop (from OpenCV) or an RGB numpy array
    result = detector.predict(face_crop)

    result = {
        "label"  : "real" | "spoof",
        "score"  : float,   # probability of being real (0-1)
        "is_real": bool
    }
    """

    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = THRESHOLD):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        candidates = [model_path]
        if str(model_path).endswith(".keras"):
            candidates.append(Path(str(model_path).replace(".keras", ".h5")))
            
        last_err = None
        self.model = None
        for path in candidates:
            if path.exists():
                logger.info(f"Loading model from: {path}")
                try:
                    self.model = tf.keras.models.load_model(str(path))
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model {path} via load_model: {e}. Trying weight-loading fallback...")
                    last_err = e
                    try:
                        from tensorflow.keras.applications import MobileNetV2
                        from tensorflow.keras import layers, models
                        base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=None)
                        x = base.output
                        x = layers.GlobalAveragePooling2D()(x)
                        x = layers.Dense(128, activation='relu')(x)
                        x = layers.Dropout(0.4)(x)
                        out = layers.Dense(1, activation='sigmoid')(x)
                        model = models.Model(base.input, out)
                        model.load_weights(str(path))
                        self.model = model
                        logger.info(f"Successfully loaded anti-spoofing weights from {path}")
                        break
                    except Exception as e2:
                        logger.error(f"Failed weight-loading fallback: {e2}")
                        last_err = e2
        if self.model is None:
            raise FileNotFoundError(
                f"Failed to load any model from candidates {candidates}. Last error: {last_err}"
            )
        self.threshold = threshold
        logger.info("Model loaded successfully.")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Accepts a BGR (OpenCV) or RGB numpy array of any size.
        Returns a preprocessed (1, IMG_SIZE, IMG_SIZE, 3) float32 array.
        Matches training pipeline: ImageDataGenerator rescale=1.0/255.
        """
        # Convert BGR -> RGB (OpenCV loads as BGR by default)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to training size
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Match training: ImageDataGenerator rescale=1.0/255
        image = image.astype(np.float32) / 255.0

        # Add batch dimension -> (1, 128, 128, 3)
        return np.expand_dims(image, axis=0)

    def predict(self, face_image: np.ndarray) -> dict:
        """
        Run liveness detection on a face crop.

        Parameters
        ----------
        face_image : np.ndarray
            A face crop as a numpy array (BGR or RGB, any size).

        Returns
        -------
        dict with keys:
            label   : "real" or "spoof"
            score   : float — probability of being a real face (0-1)
            is_real : bool
        """
        processed = self.preprocess(face_image)
        score     = float(self.model.predict(processed, verbose=0)[0][0])
        is_real   = score >= self.threshold
        label     = "real" if is_real else "spoof"

        return {
            "label"  : label,
            "score"  : round(score, 4),
            "is_real": is_real,
        }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = AntispoofingDetector()

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Could not load image: {sys.argv[1]}")
            sys.exit(1)
        result = detector.predict(img)
        print(f"Result: {result}")
    else:
        print("Usage: py -3.11 antispoofing_detector.py <path_to_image>")
        print("Or import AntispoofingDetector in your integration code.")
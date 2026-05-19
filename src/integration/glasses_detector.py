"""
Reusable glasses/sunglasses detector for verification integration.

The detector expects a cropped face image and predicts one of:
    glasses, no_glasses, sunglasses

Decision logic:
    glasses    -> allow
    no_glasses -> allow
    sunglasses -> block
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "glasses_detector_residual_cnn.keras"

CLASSES = ["glasses", "no_glasses", "sunglasses"]
DECISIONS = {
    "glasses": "allow",
    "no_glasses": "allow",
    "sunglasses": "block",
}
MESSAGES = {
    "no_glasses": "No glasses detected - verification can continue",
    "glasses": "Glasses detected - verification can continue",
    "sunglasses": "Sunglasses detected - please remove before verification",
}


class GlassesDetector:
    """Load the trained model and classify cropped face images."""

    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model_path = self.model_path.resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Glasses detector model not found: {self.model_path}")

        print(f"Loading glasses detector model: {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        self.img_size = self._get_model_input_size()
        print(f"Model input size: {self.img_size[0]}x{self.img_size[1]}")

    def predict(self, face_bgr):
        """
        Predict the glasses class for a cropped OpenCV BGR face image.

        Returns:
            label, confidence
        """
        batch = self.preprocess(face_bgr)
        probabilities = self.model.predict(batch, verbose=0)[0]

        class_index = int(np.argmax(probabilities))
        label = CLASSES[class_index]
        confidence = float(probabilities[class_index])

        return label, confidence

    def predict_with_decision(self, face_bgr):
        """Return label, confidence, decision, and status message together."""
        label, confidence = self.predict(face_bgr)

        return {
            "label": label,
            "confidence": confidence,
            "decision": self.decision(label),
            "message": self.message(label),
        }

    def decision(self, label):
        """Map model label to verification behavior."""
        if label not in DECISIONS:
            raise ValueError(f"Unknown glasses label: {label}")

        return DECISIONS[label]

    def message(self, label):
        """Map model label to a reusable integration status message."""
        if label not in MESSAGES:
            raise ValueError(f"Unknown glasses label: {label}")

        return MESSAGES[label]

    def preprocess(self, face_bgr):
        """
        Convert a cropped OpenCV BGR face image into model input.

        The trained model contains a Rescaling(1/255) layer, so this method
        keeps values in the 0-255 range and only converts to float32.
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty cropped face image received.")

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, self.img_size, interpolation=cv2.INTER_AREA)
        face_rgb = face_rgb.astype(np.float32)

        return np.expand_dims(face_rgb, axis=0)

    def _get_model_input_size(self):
        input_shape = self.model.input_shape[0] if isinstance(self.model.input_shape, list) else self.model.input_shape

        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        if height is None or width is None or channels != 3:
            raise ValueError(f"Unsupported model input shape: {input_shape}")

        return int(width), int(height)

"""
Reusable glasses/sunglasses detector for verification integration.

The detector expects a cropped face image and predicts one of:
    glasses, no_glasses, sunglasses

It can also accept a full webcam frame through predict_from_frame(frame_bgr),
where face detection and consistent cropping are handled internally.

Advisory status:
    glasses    -> allow
    no_glasses -> allow
    sunglasses -> advisory
"""

from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "glasses_detector_residual_cnn.keras"
)

CLASSES = ["glasses", "no_glasses", "sunglasses"]
DECISIONS = {
    # Keep these values for compatibility with existing integrations. The UI
    # treats glasses detection as advisory and does not stop verification.
    "glasses": "allow",
    "no_glasses": "allow",
    "sunglasses": "block",
}
MESSAGES = {
    "no_glasses": "No glasses detected. Verification continues.",
    "glasses": "Glasses detected. Verification continues.",
    "sunglasses": "Sunglasses detected. Please remove sunglasses for clearer verification.",
}
NO_FACE_RESULT = {
    "label": "no_face",
    "confidence": 0.0,
    "decision": "block",
    "message": "No face detected. Please face the camera.",
    "face_box": None,
}
SUNGLASSES_ADVISORY_CONFIDENCE_THRESHOLD = 0.90
GLASSES_STABILITY_WINDOW = 7
GLASSES_STABILITY_MIN_FRAMES = 4


class GlassesDetector:
    """Reusable wrapper around the trained glasses/sunglasses Keras model.

    The attendance system may pass either a cropped face image to
    predict_with_decision() or a full webcam frame to predict_from_frame().
    This class handles model loading, preprocessing, prediction, and the
    advisory status text needed by UI or orchestration code.
    """

    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model_path = self.model_path.resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Glasses detector model not found: {self.model_path}")

        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        # The input size is read from the saved model so callers do not need to
        # know the training image size when integrating the detector.
        self.img_size = self._get_model_input_size()
        print(f"Model input size: {self.img_size[0]}x{self.img_size[1]}")
        self.face_detector = self.build_face_detector()
        self.last_face_crop = None
        self.glasses_history = deque(maxlen=GLASSES_STABILITY_WINDOW)
        self.stable_glasses_result = None

    def predict(self, face_bgr):
        """
        Predict the glasses class for a cropped OpenCV BGR face image.

        The returned label and confidence are intentionally simple so they can
        be consumed by the verification pipeline, webcam UI, or tests.

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
        """Return prediction output together with advisory status metadata."""
        label, confidence = self.predict(face_bgr)

        # Returning message and decision here avoids duplicating status text in
        # the webcam UI or main attendance workflow.
        return {
            "label": label,
            "confidence": confidence,
            "decision": self.decision(label),
            "message": self.message(label),
        }

    def decision(self, label):
        """Map the predicted eyewear label to the existing integration status."""
        if label not in DECISIONS:
            raise ValueError(f"Unknown glasses label: {label}")

        return DECISIONS[label]

    def message(self, label):
        """Return a user-facing message for the predicted eyewear label."""
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

        # OpenCV reads webcam/crop images in BGR, while the model was trained on
        # RGB image tensors.
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Keep resizing inside the detector so every integration point uses the
        # same input dimensions as training.
        face_rgb = cv2.resize(face_rgb, self.img_size, interpolation=cv2.INTER_AREA)
        face_rgb = face_rgb.astype(np.float32)

        return np.expand_dims(face_rgb, axis=0)

    def predict_from_frame(self, frame_bgr):
        """
        Predict glasses status from a full OpenCV BGR webcam frame.

        Face detection, filtering, face selection, and margin cropping are kept
        inside this integration class so webcam callers all use the same crop.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_face_crop = None
            self.reset_frame_stability()
            return dict(NO_FACE_RESULT)

        faces = self.detect_faces(frame_bgr)
        face_box = self.select_best_face(faces, frame_bgr.shape)

        if face_box is None:
            self.last_face_crop = None
            self.reset_frame_stability()
            return dict(NO_FACE_RESULT)

        face_crop, crop_box = self.crop_face_with_margin(frame_bgr, face_box)

        if face_crop is None or face_crop.size == 0:
            self.last_face_crop = None
            self.reset_frame_stability()
            return dict(NO_FACE_RESULT)

        result = self.predict_with_decision(face_crop)
        result = self._apply_full_frame_decision(result)
        result["face_box"] = crop_box
        self.last_face_crop = face_crop

        return self._stabilize_frame_result(result)

    def reset_frame_stability(self):
        """Clear webcam-frame label stability when no usable face is present."""
        self.glasses_history.clear()
        self.stable_glasses_result = None

    def build_face_detector(self):
        """Load OpenCV's built-in frontal face Haar Cascade."""
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))

        if detector.empty():
            raise RuntimeError(f"Failed to load Haar Cascade: {cascade_path}")

        return detector

    def detect_faces(self, frame_bgr):
        """Detect candidate face boxes from a full webcam frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        return self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

    def filter_face_boxes(self, faces, frame_shape):
        """Remove Haar boxes that are unlikely to be real frontal faces."""
        frame_h, frame_w = frame_shape[:2]
        min_side = max(60, int(min(frame_w, frame_h) * 0.08))
        max_w = int(frame_w * 0.45)
        max_h = int(frame_h * 0.45)
        valid_faces = []

        for box in faces:
            x, y, w, h = [int(value) for value in box]

            if w <= 0 or h <= 0:
                continue

            aspect_ratio = w / float(h)

            if not 0.75 <= aspect_ratio <= 1.35:
                continue

            if w < min_side or h < min_side:
                continue

            if w > max_w or h > max_h:
                continue

            if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                continue

            valid_faces.append((x, y, w, h))

        return valid_faces

    def select_best_face(self, faces, frame_shape):
        """Prefer larger, central valid faces and reject frames with no valid face."""
        valid_faces = self.filter_face_boxes(faces, frame_shape)

        if not valid_faces:
            return None

        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_w * frame_h)
        frame_center_x = frame_w / 2.0
        frame_center_y = frame_h / 2.0
        max_center_distance = np.hypot(frame_center_x, frame_center_y)

        def score_face(box):
            x, y, w, h = box
            face_area_score = (w * h) / frame_area
            face_center_x = x + w / 2.0
            face_center_y = y + h / 2.0
            center_distance = np.hypot(
                face_center_x - frame_center_x,
                face_center_y - frame_center_y,
            )
            center_score = 1.0 - min(center_distance / max_center_distance, 1.0)

            return face_area_score + 0.20 * center_score

        return max(valid_faces, key=score_face)

    def crop_face_with_margin(self, frame_bgr, face_box, margin=0.18):
        """Crop a square face region with context around the selected face box."""
        x, y, w, h = [int(value) for value in face_box]
        frame_h, frame_w = frame_bgr.shape[:2]

        side = int(max(w, h) * (1.0 + 2.0 * margin))
        center_x = x + w // 2
        center_y = y + h // 2

        x1 = max(center_x - side // 2, 0)
        y1 = max(center_y - side // 2, 0)
        x2 = min(x1 + side, frame_w)
        y2 = min(y1 + side, frame_h)

        # Keep the crop square when it touches the image border.
        x1 = max(x2 - side, 0)
        y1 = max(y2 - side, 0)

        face_crop = frame_bgr[y1:y2, x1:x2]
        crop_box = (x1, y1, x2 - x1, y2 - y1)

        return face_crop, crop_box

    def _apply_full_frame_decision(self, result):
        """Use advisory wording for uncertain sunglasses results."""
        if (
            result["label"] == "sunglasses"
            and result["confidence"] < SUNGLASSES_ADVISORY_CONFIDENCE_THRESHOLD
        ):
            result["decision"] = "warning"
            result["message"] = (
                "Sunglasses detected. Please remove sunglasses for clearer verification."
            )

        return result

    def _stabilize_frame_result(self, result):
        """Return a steadier webcam result using recent full-frame predictions."""
        self.glasses_history.append(
            {
                "label": result["label"],
                "confidence": result["confidence"],
                "decision": result["decision"],
                "message": result["message"],
            }
        )

        label_counts = Counter(item["label"] for item in self.glasses_history)
        selected_label, selected_count = label_counts.most_common(1)[0]

        if (
            self.stable_glasses_result is None
            or selected_count >= GLASSES_STABILITY_MIN_FRAMES
        ):
            matching_results = [
                item for item in self.glasses_history if item["label"] == selected_label
            ]
            average_confidence = float(
                np.mean([item["confidence"] for item in matching_results])
            )
            latest_selected_result = matching_results[-1]
            self.stable_glasses_result = {
                "label": selected_label,
                "confidence": average_confidence,
                "decision": latest_selected_result["decision"],
                "message": latest_selected_result["message"],
            }

        stable_result = dict(self.stable_glasses_result)
        stable_result["face_box"] = result["face_box"]

        return stable_result

    def _get_model_input_size(self):
        """Read and validate the expected model input size from the saved model."""
        input_shape = self.model.input_shape[0] if isinstance(self.model.input_shape, list) else self.model.input_shape

        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        if height is None or width is None or channels != 3:
            raise ValueError(f"Unsupported model input shape: {input_shape}")

        return int(width), int(height)

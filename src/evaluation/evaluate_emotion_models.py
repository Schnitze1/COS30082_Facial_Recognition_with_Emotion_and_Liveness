"""
Evaluate saved emotion CNN models on the held-out AffectNet test set.

This script does not retrain or modify any saved model. It loads the finalized
vanilla and residual CNN models, evaluates them on the same folder-based test
dataset, and writes reports under reports/emotion_detection/.
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "emotion_detection" / "AffecNet"
TEST_DIR = DATA_DIR / "Test"
REPORT_DIR = PROJECT_ROOT / "reports" / "emotion_detection"

# Order identical to training/integration so prediction indices map
# to the same emotion labels in every report.
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

MODEL_SPECS = {
    "vanilla": {
        "keras": PROJECT_ROOT / "models" / "emotion_detection" / "vanilla" / "emotion_cnn_vanilla.keras",
        "h5": PROJECT_ROOT / "models" / "emotion_detection" / "vanilla" / "emotion_cnn_vanilla.h5",
    },
    "residual": {
        "keras": PROJECT_ROOT / "models" / "emotion_detection" / "residual" / "emotion_cnn_residual.keras",
        "h5": PROJECT_ROOT / "models" / "emotion_detection" / "residual" / "emotion_cnn_residual.h5",
    },
}

BATCH_SIZE = 32


def to_percent(value: float) -> float:
    """Convert a decimal metric to a percentage rounded to two decimals."""
    return round(float(value) * 100.0, 2)


def resolve_model_path(model_name: str) -> Path:
    """Resolve the saved model path without retraining or modifying the model."""
    spec = MODEL_SPECS[model_name]
    if spec["keras"].is_file():
        return spec["keras"]
    if spec["h5"].is_file():
        return spec["h5"]
    raise FileNotFoundError(
        f"No saved model found for {model_name}. Checked {spec['keras']} and {spec['h5']}."
    )


def resolve_class_folder_names(root_dir: Path) -> list[str]:
    """Resolve class folder names case-insensitively while preserving label order."""
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Test dataset folder not found: {root_dir}")

    folder_map = {
        path.name.lower().strip(): path.name
        for path in root_dir.iterdir()
        if path.is_dir()
    }
    missing = [class_name for class_name in CLASSES if class_name not in folder_map]
    if missing:
        raise FileNotFoundError(
            f"Test folder is missing expected class folder(s): {', '.join(missing)}"
        )

    return [folder_map[class_name] for class_name in CLASSES]


def get_model_input_size(model: tf.keras.Model) -> tuple[int, int]:
    """Read the saved input size so different CNN variants share one evaluator."""
    input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]
    if height is None or width is None or channels != 3:
        raise ValueError(f"Unsupported model input shape: {input_shape}")
    return int(height), int(width)


def model_has_internal_rescaling(model: tf.keras.Model) -> bool:
    """
    Detect input preprocessing layers without confusing them with BatchNormalization.

    The saved custom CNNs expect externally rescaled 0-1 input unless they
    explicitly include a Rescaling or preprocessing layer.
    """
    # BatchNormalization is not input preprocessing, so only explicit image
    # rescaling/preprocessing layers should disable external /255 scaling.
    known_preprocess_terms = ("rescaling", "preprocess", "preprocessing")
    for layer in model.layers:
        layer_name = layer.name.lower()
        class_name = layer.__class__.__name__.lower()
        if any(term in layer_name or term in class_name for term in known_preprocess_terms):
            return True
    return False


def build_test_dataset(image_size: tuple[int, int], apply_rescale: bool) -> tf.data.Dataset:
    """Create the test dataset with preprocessing matched to the loaded model."""
    class_folders = resolve_class_folder_names(TEST_DIR)
    print("Class order:", CLASSES)
    print("Resolved test folders:", class_folders)
    print("Loading test dataset from:", TEST_DIR)
    print("Image size:", image_size)
    print("Apply /255 rescale:", apply_rescale)

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=class_folders,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        # Keep shuffle disabled so y_true and y_pred remain aligned for the
        # classification report and confusion matrix.
        shuffle=False,
    )

    def preprocess(images, labels):
        """Apply only deterministic preprocessing during evaluation."""
        images = tf.cast(images, tf.float32)
        if apply_rescale:
            images = images / 255.0
        return images, labels

    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def collect_true_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect class ids in dataset order so they align with model predictions."""
    return np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in dataset])


def save_confusion_matrix(model_name: str, matrix: np.ndarray) -> None:
    """Save class-confusion results as both data and a report-ready heatmap."""
    matrix_df = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    csv_path = REPORT_DIR / f"{model_name}_confusion_matrix.csv"
    png_path = REPORT_DIR / f"{model_name}_confusion_matrix.png"

    matrix_df.to_csv(csv_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(f"{model_name.title()} CNN Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix CSV: {csv_path}")
    print(f"Saved confusion matrix PNG: {png_path}")


def save_classification_report(model_name: str, report_text: str, report_json: dict) -> None:
    """Save per-class precision, recall, and F1 metrics for later reporting."""
    text_path = REPORT_DIR / f"{model_name}_classification_report.txt"
    json_path = REPORT_DIR / f"{model_name}_classification_report.json"

    text_path.write_text(report_text, encoding="utf-8")
    json_path.write_text(json.dumps(report_json, indent=4), encoding="utf-8")

    print(f"Saved classification report TXT: {text_path}")
    print(f"Saved classification report JSON: {json_path}")


def evaluate_saved_model(model_name: str) -> dict:
    """Evaluate one trained emotion model on the common held-out test set."""
    model_path = resolve_model_path(model_name)
    print("\n" + "=" * 72)
    print(f"Evaluating {model_name} CNN")
    print("Model path:", model_path)

    model = tf.keras.models.load_model(model_path, compile=False)
    # Use the model's saved input shape so 96x96 and 128x128 models can be
    # evaluated by the same script without input-shape mismatch.
    image_size = get_model_input_size(model)

    has_internal_rescaling = model_has_internal_rescaling(model)
    if has_internal_rescaling:
        print("Detected a possible internal rescaling/preprocessing layer.")
        print("The evaluation dataset will NOT divide images by 255 again.")
    else:
        print("No internal Rescaling/preprocessing layer detected.")
        print("The evaluation dataset WILL divide images by 255, matching training.")

    test_dataset = build_test_dataset(
        image_size=image_size,
        apply_rescale=not has_internal_rescaling,
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss, accuracy = model.evaluate(test_dataset, verbose=1)
    probabilities = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = collect_true_labels(test_dataset)

    # The classification report exposes per-class precision, recall, and F1,
    # which is important when a high overall accuracy hides weak minority classes.
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        zero_division=0,
    )
    report_json = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    # The confusion matrix shows which emotion pairs are being confused, such as
    # sad/neutral or fear/surprise.
    matrix = confusion_matrix(y_true, y_pred)

    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(matrix)

    save_classification_report(model_name, report_text, report_json)
    save_confusion_matrix(model_name, matrix)

    weighted = report_json["weighted avg"]
    macro = report_json["macro avg"]
    # Percentage columns are kept alongside raw decimals to make report tables
    # easier to read without losing machine-readable metrics.
    accuracy_percent = to_percent(accuracy)
    weighted_precision_percent = to_percent(weighted["precision"])
    weighted_recall_percent = to_percent(weighted["recall"])
    weighted_f1_percent = to_percent(weighted["f1-score"])
    macro_precision_percent = to_percent(macro["precision"])
    macro_recall_percent = to_percent(macro["recall"])
    macro_f1_percent = to_percent(macro["f1-score"])

    print("\nSummary metrics:")
    print(f"Loss: {float(loss):.4f}")
    print(f"Accuracy: {accuracy_percent:.2f}%")
    print(f"Weighted precision: {weighted_precision_percent:.2f}%")
    print(f"Weighted recall: {weighted_recall_percent:.2f}%")
    print(f"Weighted F1: {weighted_f1_percent:.2f}%")
    print(f"Macro precision: {macro_precision_percent:.2f}%")
    print(f"Macro recall: {macro_recall_percent:.2f}%")
    print(f"Macro F1: {macro_f1_percent:.2f}%")

    result = {
        "model": model_name,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "input_height": image_size[0],
        "input_width": image_size[1],
        "internal_rescaling_detected": has_internal_rescaling,
        "test_loss": float(loss),
        "accuracy": float(accuracy),
        "accuracy_percent": accuracy_percent,
        "weighted_precision": float(weighted["precision"]),
        "weighted_precision_percent": weighted_precision_percent,
        "weighted_recall": float(weighted["recall"]),
        "weighted_recall_percent": weighted_recall_percent,
        "weighted_f1_score": float(weighted["f1-score"]),
        "weighted_f1_percent": weighted_f1_percent,
        "macro_precision": float(macro["precision"]),
        "macro_precision_percent": macro_precision_percent,
        "macro_recall": float(macro["recall"]),
        "macro_recall_percent": macro_recall_percent,
        "macro_f1_score": float(macro["f1-score"]),
        "macro_f1_percent": macro_f1_percent,
    }
    return result


def save_comparison_table(results: list[dict]) -> None:
    """Save side-by-side model comparison for the project report."""
    comparison_df = pd.DataFrame(results)
    csv_path = REPORT_DIR / "emotion_model_comparison.csv"
    json_path = REPORT_DIR / "emotion_model_comparison.json"

    comparison_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=4), encoding="utf-8")

    print("\nModel comparison:")
    display_columns = [
        "model",
        "test_loss",
        "accuracy_percent",
        "weighted_precision_percent",
        "weighted_recall_percent",
        "weighted_f1_percent",
        "macro_precision_percent",
        "macro_recall_percent",
        "macro_f1_percent",
    ]
    print(comparison_df[display_columns])
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison JSON: {json_path}")


def main() -> None:
    """Run evaluation for all saved emotion models without retraining."""
    print("Emotion model evaluation pipeline")
    print("Project root:", PROJECT_ROOT)
    print("Reports folder:", REPORT_DIR)
    print("No training will be performed.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_saved_model("vanilla"),
        evaluate_saved_model("residual"),
    ]
    save_comparison_table(results)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

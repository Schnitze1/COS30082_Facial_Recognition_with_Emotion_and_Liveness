"""
Evaluate the trained glasses detector.

Run from the project root:
    python src/evaluation/evaluate_glasses_detector.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "glass_detection" / "Glasses_dataset"
TEST_DIR = DATASET_DIR / "test"
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "glasses_detection"
    / "residual"
    / "glasses_detector_residual_cnn.keras"
)
REPORT_DIR = PROJECT_ROOT / "reports" / "glasses_detection"
HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "glasses_detection"
    / "residual"
    / "glasses_detector_residual_cnn_history.json"
)

# Keep this order aligned with the training script and integration detector so
# prediction indices map to the correct verification labels.
CLASSES = ["glasses", "no_glasses", "sunglasses"]
IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def to_percent(value):
    """Convert a decimal metric to a percentage rounded to two decimals."""
    return round(float(value) * 100.0, 2)


def validate_inputs():
    """Check that the trained model and held-out test split are available."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run: python src/training/glasses_detection/train_glasses_detector.py"
        )

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test folder not found: {TEST_DIR}\n"
            "Prepare the glasses dataset split before evaluating."
        )


def load_test_dataset():
    """Load the glasses test split without random augmentation or shuffling."""
    print(f"Loading test dataset from: {TEST_DIR}")
    return tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        # Deterministic ordering keeps predictions aligned with the true labels
        # used by the classification report and confusion matrix.
        shuffle=False,
    )


def save_test_metrics(test_loss, test_accuracy, report_json):
    """Save summary metrics in JSON for reuse in written reports."""
    metrics_path = REPORT_DIR / "test_metrics.json"
    weighted = report_json["weighted avg"]
    macro = report_json["macro avg"]
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "accuracy_percent": to_percent(test_accuracy),
        "weighted_precision": float(weighted["precision"]),
        "weighted_precision_percent": to_percent(weighted["precision"]),
        "weighted_recall": float(weighted["recall"]),
        "weighted_recall_percent": to_percent(weighted["recall"]),
        "weighted_f1_score": float(weighted["f1-score"]),
        "weighted_f1_percent": to_percent(weighted["f1-score"]),
        "macro_precision": float(macro["precision"]),
        "macro_precision_percent": to_percent(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "macro_recall_percent": to_percent(macro["recall"]),
        "macro_f1_score": float(macro["f1-score"]),
        "macro_f1_percent": to_percent(macro["f1-score"]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"Saved test metrics: {metrics_path}")


def collect_predictions(model, test_ds):
    """Collect model predictions and true labels in the same dataset order."""
    print("Predicting test images...")
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    return y_true, y_pred


def save_classification_report(y_true, y_pred):
    """Save precision, recall, and F1-score for each glasses class."""
    text_report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    json_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4,
        output_dict=True,
    )

    text_path = REPORT_DIR / "classification_report.txt"
    json_path = REPORT_DIR / "classification_report.json"

    # Save both text and JSON so the results are readable by humans and easy to
    # reuse in tables or scripts.
    text_path.write_text(text_report, encoding="utf-8")
    json_path.write_text(json.dumps(json_report, indent=4), encoding="utf-8")

    print("Classification report:")
    print(text_report)
    print(f"Saved report: {text_path}")
    print(f"Saved report JSON: {json_path}")
    return json_report


def save_confusion_matrix(y_true, y_pred):
    """Save class-confusion results as CSV data and a report-ready heatmap."""
    matrix = confusion_matrix(y_true, y_pred)

    csv_path = REPORT_DIR / "confusion_matrix.csv"
    png_path = REPORT_DIR / "confusion_matrix.png"

    np.savetxt(csv_path, matrix, delimiter=",", fmt="%d")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.title("Glasses Detection Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    print("Confusion matrix:")
    print(matrix)
    print(f"Saved confusion matrix CSV: {csv_path}")
    print(f"Saved confusion matrix image: {png_path}")


def save_training_curves():
    """Export training curves when history is available for report discussion."""
    if not HISTORY_PATH.is_file():
        print(f"Training history not found at {HISTORY_PATH}; skipping curves.")
        return

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    curves_path = REPORT_DIR / "accuracy_loss_curves.png"

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(curves_path, dpi=160)
    plt.close()
    print(f"Saved accuracy/loss curves: {curves_path}")


def main():
    """Evaluate the already trained glasses detector without retraining it."""
    print("=== Glasses Detector Evaluation ===")
    print(f"Model:     {MODEL_PATH}")
    print(f"Test data: {TEST_DIR}")
    print(f"Reports:   {REPORT_DIR}")
    print()

    validate_inputs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_ds = load_test_dataset()

    print("Loading trained model...")
    # Loading with compile=False avoids depending on the original training
    # optimizer state; evaluation recompiles only the metrics needed here.
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    print("Evaluating test accuracy...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {to_percent(test_accuracy):.2f}%")

    y_true, y_pred = collect_predictions(model, test_ds)
    report_json = save_classification_report(y_true, y_pred)
    print("Summary metrics:")
    print(f"Accuracy: {to_percent(test_accuracy):.2f}%")
    print(f"Weighted precision: {to_percent(report_json['weighted avg']['precision']):.2f}%")
    print(f"Weighted recall: {to_percent(report_json['weighted avg']['recall']):.2f}%")
    print(f"Weighted F1: {to_percent(report_json['weighted avg']['f1-score']):.2f}%")
    print(f"Macro precision: {to_percent(report_json['macro avg']['precision']):.2f}%")
    print(f"Macro recall: {to_percent(report_json['macro avg']['recall']):.2f}%")
    print(f"Macro F1: {to_percent(report_json['macro avg']['f1-score']):.2f}%")
    save_test_metrics(test_loss, test_accuracy, report_json)
    save_confusion_matrix(y_true, y_pred)
    save_training_curves()

    print("Evaluation complete.")


if __name__ == "__main__":
    main()

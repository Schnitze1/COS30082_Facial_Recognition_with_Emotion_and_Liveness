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
MODEL_PATH = PROJECT_ROOT / "models" / "glasses_detector_residual_cnn.keras"
REPORT_DIR = PROJECT_ROOT / "reports" / "glasses_detection"
HISTORY_PATH = REPORT_DIR / "training_history.json"

CLASSES = ["glasses", "no_glasses", "sunglasses"]
IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def validate_inputs():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run: python src/training/train_glasses_detector.py"
        )

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test folder not found: {TEST_DIR}\n"
            "Run: python src/training/split_glasses_dataset.py"
        )


def load_test_dataset():
    print(f"Loading test dataset from: {TEST_DIR}")
    return tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def save_test_metrics(test_loss, test_accuracy):
    metrics_path = REPORT_DIR / "test_metrics.json"
    metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"Saved test metrics: {metrics_path}")


def collect_predictions(model, test_ds):
    print("Predicting test images...")
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    return y_true, y_pred


def save_classification_report(y_true, y_pred):
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

    text_path.write_text(text_report, encoding="utf-8")
    json_path.write_text(json.dumps(json_report, indent=4), encoding="utf-8")

    print("Classification report:")
    print(text_report)
    print(f"Saved report: {text_path}")
    print(f"Saved report JSON: {json_path}")


def save_confusion_matrix(y_true, y_pred):
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
    print("=== Glasses Detector Evaluation ===")
    print(f"Model:     {MODEL_PATH}")
    print(f"Test data: {TEST_DIR}")
    print(f"Reports:   {REPORT_DIR}")
    print()

    validate_inputs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_ds = load_test_dataset()

    print("Loading trained model...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    print("Evaluating test accuracy...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    save_test_metrics(test_loss, test_accuracy)

    y_true, y_pred = collect_predictions(model, test_ds)
    save_classification_report(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred)
    save_training_curves()

    print("Evaluation complete.")


if __name__ == "__main__":
    main()

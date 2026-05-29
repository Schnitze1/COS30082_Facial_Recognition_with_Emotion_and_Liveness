"""
Evaluate trained EfficientNet-B0 emotion model on AffectNet test set.
Outputs: classification report (txt + JSON), confusion matrix (CSV + PNG).
Usage: python evaluate_emotion_efficientnet.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

#  Config 
CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DATASET_TEST = "data/Test"
MODEL_DIR = "models/emotion_detection_2/efficientnet"
MODEL_NAME = "emotion_efficientnet"
REPORT_DIR = "reports/emotion_detection_2"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = len(CLASSES)
AUTOTUNE = tf.data.AUTOTUNE


# Data   
def collect_images(root):
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            print(f"  [warn] '{cls}' not found in {root}")
            continue
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder, f))
                labels.append(idx)
    return np.array(paths), np.array(labels)


def parse_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label


def build_test_dataset():
    paths, labels = collect_images(DATASET_TEST)
    print(f"Test samples: {len(paths)}")
    one_hot = tf.one_hot(labels, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, one_hot))
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, labels


# ── Evaluation ──────────────────────────────────────────────
def evaluate():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load model
    keras_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
    h5_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.h5")
    model_path = keras_path if os.path.exists(keras_path) else h5_path
    print(f"Loading: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    test_ds, true_labels = build_test_dataset()

    # Loss + accuracy
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"  Loss: {loss:.4f}  Accuracy: {acc:.4f}")

    # Predictions
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = true_labels[:len(y_pred)]

    # Classification report
    report_txt = classification_report(y_true, y_pred, target_names=CLASSES)
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    report_dict["test_loss"] = float(loss)
    report_dict["test_accuracy"] = float(acc)
    print(f"\n{report_txt}")

    with open(os.path.join(REPORT_DIR, "efficientnet_report.txt"), "w") as f:
        f.write(f"Model: {MODEL_NAME}\n\n{report_txt}\nLoss: {loss:.4f}  Accuracy: {acc:.4f}\n")
    with open(os.path.join(REPORT_DIR, "efficientnet_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(REPORT_DIR, "efficientnet_confusion_matrix.csv"))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"EfficientNet-B0 — Confusion Matrix (Acc: {acc:.4f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "efficientnet_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"\nReports saved to {REPORT_DIR}/")
    return report_dict


if __name__ == "__main__":
    evaluate()
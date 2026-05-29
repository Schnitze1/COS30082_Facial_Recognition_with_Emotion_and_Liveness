"""
MobileNetV2 fine-tuned on CelebA-Spoof (binary: real vs spoof).
Uses Protocol 2 (test_on_low_quality_device) for webcam generalisation.
Usage: run as Kaggle notebook with CelebA-Spoof dataset attached.
"""
 
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
 
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
 
 
# ── Config ──────────────────────────────────────────────────────────────────────
CELEBA_ROOT           = Path("/kaggle/input/datasets/mabdullahsajid/celeba-spoofing/CelebA_Spoof")
DATA_ROOT             = CELEBA_ROOT
MODEL_DIR             = Path("/kaggle/working/models/anti_spoofing")
REPORT_DIR            = Path("/kaggle/working/reports/anti_spoofing")
TRAIN_JSON            = CELEBA_ROOT / "metas" / "protocol2" / "test_on_low_quality_device" / "train_label.json"
TEST_JSON             = CELEBA_ROOT / "metas" / "protocol2" / "test_on_low_quality_device" / "test_label.json"
IMG_SIZE              = 128
BATCH_SIZE            = 32
EPOCHS_PHASE1         = 15
EPOCHS_PHASE2         = 30
LR_PHASE1             = 1e-4
LR_PHASE2             = 1e-5
FINE_TUNE_AT          = 100    # unfreeze MobileNetV2 layers from this index onwards
MAX_SAMPLES_PER_CLASS = 50000
 
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH       = MODEL_DIR / "antispoofing_model.h5"
MODEL_PATH_KERAS = MODEL_DIR / "antispoofing_model.keras"
 
 
# ── Data loading ────────────────────────────────────────────────────────────────
def load_annotations(json_path, data_root, max_per_class=None):
    """
    Parses CelebA-Spoof annotation JSON into balanced image path / label lists.
 
    :param json_path: Path to the annotation JSON file.
    :param data_root: Root directory of the CelebA-Spoof dataset.
    :param max_per_class: Optional cap on samples per class for balanced training.
    :return: Tuple of (image_paths, labels) — shuffled and balanced.
    """
    with open(json_path, "r") as f:
        annotations = json.load(f)
 
    real_paths, fake_paths = [], []
    for rel_path, attrs in annotations.items():
        full_path = str(data_root / rel_path)
        if int(attrs[43]) == 0:   # 0 = live/real in CelebA-Spoof
            real_paths.append(full_path)
        else:                      # 1 = spoof/fake
            fake_paths.append(full_path)
 
    if max_per_class is not None:
        np.random.seed(42)
        real_paths = list(np.random.choice(real_paths, min(max_per_class, len(real_paths)), replace=False))
        fake_paths = list(np.random.choice(fake_paths, min(max_per_class, len(fake_paths)), replace=False))
 
    image_paths = real_paths + fake_paths
    labels      = [1.0] * len(real_paths) + [0.0] * len(fake_paths)
 
    combined = list(zip(image_paths, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    image_paths, labels = zip(*combined)
 
    print(f"  Real: {len(real_paths)}  Fake: {len(fake_paths)}  Total: {len(image_paths)}")
    return list(image_paths), list(labels)
 
 
def preprocess_image(path, label, augment=False):
    """
    Decodes, resizes, and optionally augments a single image.
 
    :param path: String path to the image file.
    :param label: Binary label (1.0 = real, 0.0 = spoof).
    :param augment: If True, applies random augmentations for training.
    :return: Tuple of (preprocessed image tensor, label).
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0   # normalise to [0, 1]
 
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = img + tf.random.normal(tf.shape(img), stddev=0.02)
        img = tf.clip_by_value(img, 0.0, 1.0)
 
    return img, label
 
 
def build_dataset(image_paths, labels, augment=False):
    """
    Constructs a tf.data pipeline for efficient batched training or evaluation.
 
    :param image_paths: List of image file paths.
    :param labels: List of binary labels corresponding to image_paths.
    :param augment: If True, applies data augmentation (training only).
    :return: Batched and prefetched tf.data.Dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(lambda p, l: preprocess_image(p, l, augment=augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 10000))
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
 
 
# ── Load data ───────────────────────────────────────────────────────────────────
print("Loading training annotations...")
train_paths, train_labels = load_annotations(TRAIN_JSON, DATA_ROOT, MAX_SAMPLES_PER_CLASS)
split        = int(len(train_paths) * 0.9)
val_paths,   val_labels   = train_paths[split:], train_labels[split:]
train_paths, train_labels = train_paths[:split], train_labels[:split]
 
print("Loading test annotations...")
test_paths, test_labels = load_annotations(TEST_JSON, DATA_ROOT, max_per_class=10000)
 
train_ds = build_dataset(train_paths, train_labels, augment=True)
val_ds   = build_dataset(val_paths,   val_labels)
test_ds  = build_dataset(test_paths,  test_labels)
 
class_weights_array = compute_class_weight("balanced", classes=np.array([0.0, 1.0]),
                                            y=np.array(train_labels))
class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
print(f"Class weights: {class_weights}")
 
 
# ── Model ───────────────────────────────────────────────────────────────────────
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False   # freeze backbone for phase 1
 
x      = GlobalAveragePooling2D()(base_model.output)
x      = Dense(128, activation="relu")(x)
x      = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)   # binary output: real vs spoof
 
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
 
 
# ── Callbacks ───────────────────────────────────────────────────────────────────
def get_callbacks():
    """
    Returns standard training callbacks: checkpoint, early stopping, LR reduction.
 
    :return: List of Keras callback instances.
    """
    return [
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
 
 
# ── Phase 1: Train head only ────────────────────────────────────────────────────
print("\n=== Phase 1: Training classification head ===")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1,
                     class_weight=class_weights, callbacks=get_callbacks())
 
# ── Phase 2: Fine-tune top MobileNetV2 layers ───────────────────────────────────
print(f"\n=== Phase 2: Fine-tuning from layer {FINE_TUNE_AT} onwards ===")
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False   # keep lower layers frozen
 
model.compile(optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
              loss="binary_crossentropy", metrics=["accuracy"])
history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2,
                     class_weight=class_weights, callbacks=get_callbacks())
 
model.save(str(MODEL_PATH_KERAS))
print(f"Saved: {MODEL_PATH_KERAS}")
 
 
# ── Evaluation ──────────────────────────────────────────────────────────────────
model.load_weights(str(MODEL_PATH))
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}  Loss: {test_loss:.4f}")
 
y_true  = np.array(test_labels)
y_pred  = model.predict(test_ds, verbose=1).flatten()
y_label = (y_pred >= 0.5).astype(int)
print(classification_report(y_true, y_label, target_names=["fake", "real"]))
 
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc     = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")
 
 
# ── Save plots and history ──────────────────────────────────────────────────────
acc      = history1.history["accuracy"]     + history2.history["accuracy"]
val      = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss     = history1.history["loss"]         + history2.history["loss"]
val_loss = history1.history["val_loss"]     + history2.history["val_loss"]
 
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train"); plt.plot(val, label="Val")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label="Train"); plt.plot(val_loss, label="Val")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "training_curves.png")
 
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}"); plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Anti-Spoofing"); plt.legend()
plt.savefig(REPORT_DIR / "roc_curve.png")
 
cm = confusion_matrix(y_true, y_label)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["fake", "real"], yticklabels=["fake", "real"])
plt.title("Confusion Matrix - Anti-Spoofing"); plt.ylabel("True"); plt.xlabel("Predicted")
plt.savefig(REPORT_DIR / "confusion_matrix.png")
 
with open(REPORT_DIR / "training_history.json", "w") as f:
    json.dump({"accuracy": acc, "val_accuracy": val, "loss": loss, "val_loss": val_loss,
               "auc": float(roc_auc), "test_accuracy": float(test_acc)}, f, indent=2)
 
print("\nTraining complete!")

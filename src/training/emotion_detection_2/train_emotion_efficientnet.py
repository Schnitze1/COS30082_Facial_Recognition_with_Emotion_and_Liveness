"""
EfficientNet-B0 fine-tuned on AffectNet (8 emotions).
Optimised for speed (tf.data, mixed precision) and accuracy (224x224, two-phase training).
Usage: python train_emotion_efficientnet.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Mixed precision (~2x faster on modern GPUs) ────────────
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ── Config ──────────────────────────────────────────────────
DATASET_ROOT = "/kaggle/input/datasets/mstjebashazida/affectnet/archive (3)"
CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = 224  # native EfficientNet-B0 resolution
BATCH_SIZE = 64
EPOCHS_PHASE1 = 5   # head-only warmup
EPOCHS_PHASE2 = 30  # fine-tune
LR_PHASE1 = 1e-3
LR_PHASE2 = 3e-4
FREEZE_RATIO = 0.50
VAL_SPLIT = 0.20
MODEL_DIR = "/kaggle/working/models"
MODEL_NAME = "emotion_efficientnet"
AUTOTUNE = tf.data.AUTOTUNE


# ── Data loading (tf.data) ──────────────────────────────────
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


def parse_image(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)  # keep 0-255 range

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15 * 255)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        # random crop for slight zoom/shift
        img = tf.image.resize(img, [IMAGE_SIZE + 20, IMAGE_SIZE + 20])
        img = tf.image.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])

    img = tf.clip_by_value(img, 0.0, 255.0)
    # EfficientNet's own preprocessing (expects 0-255 input)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label


def _parse_train(path, label):
    return parse_image(path, label, augment=True)

def _parse_val(path, label):
    return parse_image(path, label, augment=False)


def make_dataset(paths, labels, augment=False, shuffle=False):
    one_hot = tf.one_hot(labels, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, one_hot))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    parse_fn = _parse_train if augment else _parse_val
    ds = ds.map(parse_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def load_data():
    train_paths, train_labels = collect_images(os.path.join(DATASET_ROOT, "Train"))
    test_paths, test_labels = collect_images(os.path.join(DATASET_ROOT, "Test"))
    print(f"Dataset — train: {len(train_paths)}  test: {len(test_paths)}")

    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=VAL_SPLIT, stratify=train_labels, random_state=42
    )
    print(f"Split   — train: {len(tr_paths)}  val: {len(val_paths)}")

    weights = compute_class_weight("balanced", classes=np.unique(tr_labels), y=tr_labels)
    class_weights = dict(enumerate(weights.tolist()))

    train_ds = make_dataset(tr_paths, tr_labels, augment=True, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels)
    test_ds = make_dataset(test_paths, test_labels)

    return train_ds, val_ds, test_ds, class_weights


# ── Model ───────────────────────────────────────────────────
def build_model(trainable_backbone=False):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    if trainable_backbone:
        n_freeze = int(len(base.layers) * FREEZE_RATIO)
        for layer in base.layers[:n_freeze]:
            layer.trainable = False
        for layer in base.layers[n_freeze:]:
            layer.trainable = True
        print(f"Backbone: {n_freeze} frozen, {len(base.layers) - n_freeze} trainable")
    else:
        base.trainable = False
        print("Backbone: fully frozen (phase 1 warmup)")

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # float32 output for numerical stability with mixed precision
    x = layers.Dense(NUM_CLASSES, dtype="float32")(x)
    x = layers.Activation("softmax", dtype="float32")(x)

    return models.Model(base.input, x, name="emotion_efficientnet")


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
    )


# ── Training ────────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]


def main():
    # ── GPU check ───────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("⚠ No GPU found — training will be very slow on CPU!")
        print("  On Kaggle: Settings → Accelerator → GPU T4 x2")

    os.makedirs(MODEL_DIR, exist_ok=True)
    train_ds, val_ds, test_ds, class_weights = load_data()

    # ── Phase 1: train head only (fast warmup) ──────────────
    print("\n" + "=" * 50)
    print("Phase 1: Head-only warmup")
    print("=" * 50)
    model = build_model(trainable_backbone=False)
    compile_model(model, LR_PHASE1)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1,
              class_weight=class_weights, verbose=1)

    # ── Phase 2: unfreeze backbone and fine-tune ────────────
    print("\n" + "=" * 50)
    print("Phase 2: Fine-tuning backbone")
    print("=" * 50)
    n_freeze = int(len(model.layers) * FREEZE_RATIO)
    for layer in model.layers[:n_freeze]:
        layer.trainable = False
    for layer in model.layers[n_freeze:]:
        if not isinstance(layer, layers.BatchNormalization):  # keep BN frozen
            layer.trainable = True
    compile_model(model, LR_PHASE2)

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2,
                        class_weight=class_weights, callbacks=get_callbacks(), verbose=1)

    # ── Save ────────────────────────────────────────────────
    model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras"))
    model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}.h5"))
    with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # ── Evaluate ────────────────────────────────────────────
    print("\n--- Test evaluation ---")
    for name, val in zip(model.metrics_names, model.evaluate(test_ds)):
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
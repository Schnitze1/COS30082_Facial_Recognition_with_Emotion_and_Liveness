"""
Hybrid CNN-Transformer emotion detector on AffectNet (8 emotions).
CNN stem extracts spatial features, Transformer encoder captures long-range dependencies.
Usage: python train_emotion_hybrid_transformer.py
"""

import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Config ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_ROOT = os.environ.get(
    "AFFECTNET_DATA_DIR",
    os.path.join(PROJECT_ROOT, "data", "AffectNet")
)
CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = 96
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 60))
BASE_LR = float(os.environ.get("TRAIN_LR", 1e-4))
MIN_LR = 1e-7
WARMUP_EPOCHS = int(os.environ.get("TRAIN_WARMUP_EPOCHS", 5))
VAL_SPLIT = 0.20
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "emotion_detection_2", "hybrid_transformer")
MODEL_NAME = "emotion_hybrid_transformer"
AUTOTUNE = tf.data.AUTOTUNE

# Transformer
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.10
NUM_PATCHES = 12 * 12  # after 3 pooling layers: 96 -> 48 -> 24 -> 12


# ── LR schedule: linear warmup → cosine decay ──────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-7):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        progress = tf.maximum(0.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr": self.min_lr}


# ── Custom layers ───────────────────────────────────────────
class PatchPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kw):
        super().__init__(**kw)
        self.pos_embed = layers.Embedding(num_patches, embed_dim)

    def call(self, x):
        return x + self.pos_embed(tf.range(tf.shape(x)[1]))


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kw):
        super().__init__(**kw)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        normed = self.norm1(x)
        x = x + self.drop1(self.attn(normed, normed, training=training), training=training)
        normed = self.norm2(x)
        x = x + self.drop2(self.ffn(normed), training=training)
        return x


# ── Data loading (tf.data) ─────────────────────────────────
def collect_images(root):
    # Auto-resolve Train subdirectory if subclasses are not directly in root
    if not any(os.path.isdir(os.path.join(root, c)) for c in CLASSES):
        train_root = os.path.join(root, "Train")
        if os.path.isdir(train_root):
            root = train_root
            print(f"Redirecting image collection root to: {root}")
            
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
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.resize(img, [IMAGE_SIZE + 10, IMAGE_SIZE + 10])
        img = tf.image.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])

    return tf.clip_by_value(img, 0.0, 1.0), label


def _parse_train(p, l): return parse_image(p, l, augment=True)
def _parse_val(p, l):   return parse_image(p, l, augment=False)


def make_dataset(paths, labels, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, tf.one_hot(labels, NUM_CLASSES)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(_parse_train if augment else _parse_val, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def load_data():
    train_paths, train_labels = collect_images(os.path.join(DATASET_ROOT, "Train"))
    test_paths, test_labels = collect_images(os.path.join(DATASET_ROOT, "Test"))
    print(f"Dataset — train: {len(train_paths)}  test: {len(test_paths)}")

    tr_p, val_p, tr_l, val_l = train_test_split(
        train_paths, train_labels, test_size=VAL_SPLIT, stratify=train_labels, random_state=42
    )
    print(f"Split   — train: {len(tr_p)}  val: {len(val_p)}")

    weights = compute_class_weight("balanced", classes=np.unique(tr_l), y=tr_l)
    class_weights = dict(enumerate(weights.tolist()))

    train_ds = make_dataset(tr_p, tr_l, augment=True, shuffle=True)
    val_ds = make_dataset(val_p, val_l)
    test_ds = make_dataset(test_paths, test_labels)
    return train_ds, val_ds, test_ds, class_weights, len(tr_p) // BATCH_SIZE


# ── Model ───────────────────────────────────────────────────
def cnn_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_model():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # CNN stem → (12, 12, 128)
    x = cnn_block(inputs, 32, "cnn1")
    x = cnn_block(x, 64, "cnn2")
    x = cnn_block(x, 128, "cnn3")

    # Reshape to token sequence → (144, 128)
    x = layers.Reshape((NUM_PATCHES, 128))(x)
    x = layers.Dense(EMBED_DIM)(x)
    x = PatchPositionEmbedding(NUM_PATCHES, EMBED_DIM)(x)
    x = layers.Dropout(DROPOUT)(x)

    # Transformer encoder
    for i in range(NUM_TRANSFORMER_LAYERS):
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT, name=f"transformer_{i}")(x)

    # Pool + classify
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="emotion_hybrid_transformer")


# ── Training ────────────────────────────────────────────────
def main():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU(s): {[g.name for g in gpus]}" if gpus else "⚠ No GPU found")

    os.makedirs(MODEL_DIR, exist_ok=True)
    train_ds, val_ds, test_ds, class_weights, steps_per_epoch = load_data()

    model = build_model()
    model.summary()

    lr_schedule = WarmupCosineDecay(
        base_lr=BASE_LR,
        warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
        total_steps=EPOCHS * steps_per_epoch,
        min_lr=MIN_LR,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
    )

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras"),
                            monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
    )

    # Save
    model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras"))
    model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}.h5"))
    with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # Evaluate
    print("\n--- Test evaluation ---")
    for name, val in zip(model.metrics_names, model.evaluate(test_ds)):
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
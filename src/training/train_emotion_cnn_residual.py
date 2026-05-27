"""
src/training/emotion_detection/train_emotion_cnn_residual.py

Complete training script for AffectNet (8 classes) from scratch.
Architecture: 128x128 residual CNN with light SE, cosine LR, label smoothing,
and strong tf.data augmentation. No pre-trained weights.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix

try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    try:
        from tensorflow.keras.optimizers.experimental import AdamW
    except ImportError:
        AdamW = None

np.random.seed(42)
tf.random.set_seed(42)


# 1. Config
class Config:
    """Configuration for the residual emotion CNN experiment.

    This model is saved separately from the vanilla CNN so both architectures
    can be compared without overwriting earlier trained weights.
    """

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.environ.get(
        "AFFECTNET_DATA_DIR",
        os.path.join(PROJECT_ROOT, "data", "AffectNet")
    )
    TRAIN_DIR = os.path.join(DATA_DIR, "Train")
    TEST_DIR = os.path.join(DATA_DIR, "Test")

    IMG_SIZE = (128, 128)
    BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 32))
    EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 120))
    LEARNING_RATE = float(os.environ.get("TRAIN_LR", 3e-4))
    MIN_LEARNING_RATE = 1e-6
    EARLY_STOPPING_PATIENCE = 25
    MODEL_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection", "residual", "emotion_cnn_residual.keras"
    )
    H5_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection", "residual", "emotion_cnn_residual.h5"
    )
    HISTORY_SAVE_PATH = os.path.join(
        PROJECT_ROOT,
        "models",
        "emotion_detection",
        "residual",
        "emotion_cnn_residual_history.json",
    )

    # The class order must stay fixed because it defines both dataset label
    # encoding and the order of the model's softmax outputs.
    CLASSES = [
        "anger", "contempt", "disgust", "fear",
        "happy", "neutral", "sad", "surprise"
    ]

    @classmethod
    def print_info(cls):
        """Print path diagnostics before TensorFlow starts loading image data."""
        print("=== AffectNet Best Model Training ===")
        print("PROJECT_ROOT:", cls.PROJECT_ROOT)
        print("DATA_DIR:", cls.DATA_DIR)
        print("TRAIN_DIR:", cls.TRAIN_DIR)
        print("TEST_DIR:", cls.TEST_DIR)
        print("Train exists:", os.path.isdir(cls.TRAIN_DIR))
        print("Test exists:", os.path.isdir(cls.TEST_DIR))
        print("IMG_SIZE:", cls.IMG_SIZE)


# 2. Building blocks
def se_block(x, reduction=8, name=None):
    """Apply lightweight channel attention to recalibrate feature maps."""
    channels = int(x.shape[-1])
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(channels // reduction, 4), activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, channels))(s)
    return layers.Multiply(name=name)([x, s])


def residual_block(x, filters, stride=1, use_se=False, l2_reg=None, name=None):
    """Build a residual block that preserves information through the skip path."""
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                      kernel_regularizer=l2_reg, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False,
                      kernel_regularizer=l2_reg, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Projection is required whenever the shortcut shape no longer matches the
    # main path because of a stride change or a different channel count.
    if int(shortcut.shape[-1]) != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False,
                                 kernel_regularizer=l2_reg, name=f"{name}_sc")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_sc_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    if use_se:
        x = se_block(x, name=f"{name}_se")
    return x


# 3. Model
class BestEmotionCNN:
    """Residual CNN trained from scratch for stronger emotion feature learning."""

    def __init__(self, config):
        self.config = config
        self.model = self._build()

    def _build(self):
        """Construct the residual CNN while keeping the parameter count controlled."""
        cfg = self.config
        l2_reg = regularizers.l2(1e-4) if AdamW is None else None
        inputs = layers.Input(shape=(*cfg.IMG_SIZE, 3))

        # The stem downsamples 128x128 crops early so later residual stages can
        # learn richer features without excessive memory use.
        x = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False,
                          kernel_regularizer=l2_reg)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = residual_block(x, 64, stride=1, use_se=False, l2_reg=l2_reg, name="s1_1")
        x = residual_block(x, 64, stride=1, use_se=False, l2_reg=l2_reg, name="s1_2")

        x = residual_block(x, 128, stride=2, use_se=False, l2_reg=l2_reg, name="s2_1")
        x = residual_block(x, 128, stride=1, use_se=False, l2_reg=l2_reg, name="s2_2")

        x = residual_block(x, 256, stride=2, use_se=True, l2_reg=l2_reg, name="s3_1")
        x = residual_block(x, 256, stride=1, use_se=True, l2_reg=l2_reg, name="s3_2")

        x = residual_block(x, 256, stride=2, use_se=True, l2_reg=l2_reg, name="s4_1")

        x = layers.GlobalAveragePooling2D()(x)  # Summarises spatial features without flattening thousands of activations.
        x = layers.Dense(256, use_bias=False, kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(len(cfg.CLASSES), activation="softmax")(x)

        return models.Model(inputs, outputs, name="BestEmotionCNN")

    def get_model(self):
        return self.model


# 4. Data pipeline (tf.data — faster than ImageDataGenerator)
def resolve_class_folder_names(root_dir, classes):
    """Resolve class folders case-insensitively while preserving label order."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    folder_map = {
        name.lower().strip(): name
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    }
    missing = [class_name for class_name in classes if class_name not in folder_map]
    if missing:
        raise FileNotFoundError(
            f"Missing class folder(s) in {root_dir}: {', '.join(missing)}"
        )
    return [folder_map[class_name] for class_name in classes]


def load_datasets(cfg):
    """Load train, validation, and test datasets with matched preprocessing."""
    for directory in (cfg.TRAIN_DIR, cfg.TEST_DIR):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

    train_class_folders = resolve_class_folder_names(cfg.TRAIN_DIR, cfg.CLASSES)
    test_class_folders = resolve_class_folder_names(cfg.TEST_DIR, cfg.CLASSES)

    print("Train class folders:", train_class_folders)
    print("Test class folders:", test_class_folders)
    print("Loading train/validation split (80/20)...")

    # batch_size=None allows augmentation to run per image before batching.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=train_class_folders,
        image_size=cfg.IMG_SIZE,
        batch_size=None,
        validation_split=0.2,
        subset="training",
        seed=42,
        shuffle=True,
    )
    # Validation keeps shuffle disabled so metric reporting remains deterministic.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=train_class_folders,
        image_size=cfg.IMG_SIZE,
        batch_size=None,
        validation_split=0.2,
        subset="validation",
        seed=42,
        shuffle=False,
    )

    print("Loading test data...")
    # Test order must remain stable for the classification report and confusion matrix.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=test_class_folders,
        image_size=cfg.IMG_SIZE,
        batch_size=None,
        shuffle=False,
    )

    IMG_H, IMG_W = cfg.IMG_SIZE

    def augment(image, label):
        """Apply augmentation that improves robustness without changing the expression label."""
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.resize_with_crop_or_pad(image, IMG_H + 16, IMG_W + 16)
        image = tf.image.random_crop(image, size=(IMG_H, IMG_W, 3))
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def preprocess(image, label):
        """Apply the same pixel scaling used by training without random transforms."""
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = (
        train_ds
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


# 5. Main
if __name__ == "__main__":
    config = Config()
    config.print_info()

    train_ds, val_ds, test_ds = load_datasets(config)

    cnn = BestEmotionCNN(config)
    model = cnn.get_model()

    print("Calculating training steps...")
    train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    if train_batches == tf.data.experimental.UNKNOWN_CARDINALITY:
        train_batches = sum(1 for _ in train_ds)
    total_steps = int(train_batches) * config.EPOCHS
    alpha = config.MIN_LEARNING_RATE / config.LEARNING_RATE if config.LEARNING_RATE > 0 else 0.0

    # Cosine decay gradually lowers the learning rate, which helps fine detail
    # in facial expression features settle after the early high-learning phase.
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=total_steps,
        alpha=alpha,
    )

    if AdamW is not None:
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
        print("Optimizer: AdamW (weight_decay=1e-4) + CosineDecay")
    else:
        print("Optimizer: Adam + L2(1e-4) + CosineDecay")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        # Label smoothing reduces overconfidence on ambiguous emotion labels.
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        ],
    )
    model.summary()

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    cb = [
        # Checkpointing stores the best validation-loss model instead of assuming
        # the final epoch is the most generalisable one.
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=cb,
        verbose=1,
    )

    print("\nEvaluating on test data...")
    test_loss, test_acc, test_top2 = model.evaluate(test_ds, verbose=1)
    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test top-2 acc: {test_top2:.4f}")

    y_pred = model.predict(test_ds, verbose=1)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])

    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls, target_names=config.CLASSES))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_cls, y_pred_cls))

    model.save(config.MODEL_SAVE_PATH)
    print(f"\nSaved Keras model to {config.MODEL_SAVE_PATH}")
    try:
        model.save(config.H5_SAVE_PATH)
        print(f"Saved H5 model to {config.H5_SAVE_PATH}")
    except Exception as exc:
        print(f"Could not save H5 model: {exc}")

    os.makedirs(os.path.dirname(config.HISTORY_SAVE_PATH), exist_ok=True)
    with open(config.HISTORY_SAVE_PATH, "w", encoding="utf-8") as history_file:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            history_file,
            indent=4,
        )
    print(f"\nTraining history saved to {config.HISTORY_SAVE_PATH}")
    print("Done.")

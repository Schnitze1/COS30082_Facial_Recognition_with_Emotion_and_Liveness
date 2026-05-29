"""
Train a from-scratch residual CNN for glasses/sunglasses detection.

Run from the project root:
    
    python src/training/glasses_detection/train_glasses_detector.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Configuration
class Config:
    """Training configuration for the residual glasses detector.

    The dataset paths and model export paths are grouped here so the training,
    evaluation, and integration code can refer to the same organised structure.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATASET_DIR = PROJECT_ROOT / "data" / "glass_detection" / "Glasses_dataset"
    TRAIN_DIR = DATASET_DIR / "train"
    VAL_DIR = DATASET_DIR / "val"
    TEST_DIR = DATASET_DIR / "test"

    MODEL_DIR = PROJECT_ROOT / "models" / "glasses_detection" / "residual"
    REPORT_DIR = PROJECT_ROOT / "reports" / "glasses_detection"

    KERAS_MODEL_PATH = MODEL_DIR / "glasses_detector_residual_cnn.keras"
    H5_MODEL_PATH = MODEL_DIR / "glasses_detector_residual_cnn.h5"
    HISTORY_PATH = MODEL_DIR / "glasses_detector_residual_cnn_history.json"
    CURVES_PATH = REPORT_DIR / "training_curves.png"

    # Keep this order fixed so the detector's output index maps consistently to
    # the verification decisions used later in the attendance pipeline.
    CLASSES = ["glasses", "no_glasses", "sunglasses"]
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3


# Dataset Validation and Loading
def validate_split_dataset():
    """Fail early if a train/validation/test split is missing or incomplete."""
    for split_dir in [Config.TRAIN_DIR, Config.VAL_DIR, Config.TEST_DIR]:
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}\n"
                "Run the glasses dataset split step before training."
            )

        for class_name in Config.CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")


def load_datasets():
    """Load image datasets using the same class order and image size as the model."""
    print("Loading train, validation, and test datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(Config.TRAIN_DIR),
        labels="inferred",
        label_mode="categorical",
        class_names=Config.CLASSES,
        image_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    # Validation and test datasets are not shuffled so repeated evaluations
    # remain comparable and confusion matrices align with the same sample order.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(Config.VAL_DIR),
        labels="inferred",
        label_mode="categorical",
        class_names=Config.CLASSES,
        image_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(Config.TEST_DIR),
        labels="inferred",
        label_mode="categorical",
        class_names=Config.CLASSES,
        image_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


# Model Architecture
def conv_block(x, filters, name):
    """Extract local visual features while reducing spatial resolution."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.MaxPooling2D(pool_size=2, name=f"{name}_pool")(x)
    return x


def residual_block(x, filters, name):
    """Use a skip connection so deeper layers can refine features without discarding them."""
    shortcut = x

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_shortcut")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def build_model():
    """Construct the from-scratch glasses detector used for verification."""
    print("Building CNN from scratch. No pretrained model is used.")

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.04, seed=SEED),
            layers.RandomZoom(0.08, seed=SEED),
            layers.RandomContrast(0.10, seed=SEED),
        ],
        name="training_augmentation",
    )

    inputs = layers.Input(shape=(128, 128, 3), name="face_crop")

    # Rescaling is inside the model so webcam integration can pass normal 0-255
    # face crops without duplicating preprocessing logic.
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="initial_conv32")(x)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.Activation("relu", name="initial_relu")(x)
    x = layers.MaxPooling2D(pool_size=2, name="initial_pool")(x)

    x = conv_block(x, 64, "conv64")
    x = residual_block(x, 64, "residual64")

    x = conv_block(x, 128, "conv128")
    x = residual_block(x, 128, "residual128")

    x = conv_block(x, 256, "conv256")

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)  # Reduces feature maps without a large flattening layer.
    x = layers.Dense(256, use_bias=False, name="dense256")(x)
    x = layers.BatchNormalization(name="dense256_bn")(x)
    x = layers.Activation("relu", name="dense256_relu")(x)
    x = layers.Dropout(0.4, name="dropout_0_4")(x)
    outputs = layers.Dense(3, activation="softmax", name="class_output")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="GlassesResidualCNN")


# Class Weighting and History Export
def compute_class_weights():
    """Compute balanced class weights from training-folder frequencies."""
    labels = []
    print("Checking class balance from training folder...")

    for class_index, class_name in enumerate(Config.CLASSES):
        class_dir = Config.TRAIN_DIR / class_name
        image_count = sum(1 for path in class_dir.iterdir() if path.is_file())
        labels.extend([class_index] * image_count)
        print(f"{class_name:12s}: {image_count} training images")

    if not labels:
        raise ValueError("No training images found.")

    # Balanced weights reduce bias toward whichever eyewear class has more
    # training images without manually choosing class importance.
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(Config.CLASSES)),
        y=np.array(labels),
    )
    class_weights = {index: float(weight) for index, weight in enumerate(weights)}
    print(f"Class weights: {class_weights}")
    return class_weights


def save_training_history(history):
    """Save training curves for reporting and later comparison with test metrics."""
    Config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    Config.HISTORY_PATH.write_text(json.dumps(history_data, indent=4), encoding="utf-8")
    print(f"Saved training history: {Config.HISTORY_PATH}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(Config.CURVES_PATH, dpi=160)
    plt.close()
    print(f"Saved training curves: {Config.CURVES_PATH}")


# Training Entry Point
def main():
    print("=== Glasses Detector Training ===")
    print(f"Dataset train: {Config.TRAIN_DIR}")
    print(f"Dataset val:   {Config.VAL_DIR}")
    print(f"Dataset test:  {Config.TEST_DIR}")
    print(f"Classes:       {Config.CLASSES}")
    print()

    validate_split_dataset()
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_datasets()
    class_weights = compute_class_weights()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    training_callbacks = [
        # Early stopping and checkpointing keep the best validation model rather
        # than the last epoch, which may have started to overfit.
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Config.KERAS_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=training_callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print("Evaluating best restored weights on the test set...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("Saving final model files...")
    model.save(str(Config.KERAS_MODEL_PATH))
    print(f"Saved Keras model: {Config.KERAS_MODEL_PATH}")

    model.save(str(Config.H5_MODEL_PATH))
    print(f"Saved H5 model:    {Config.H5_MODEL_PATH}")

    save_training_history(history)
    print("Training complete.")


if __name__ == "__main__":
    main()

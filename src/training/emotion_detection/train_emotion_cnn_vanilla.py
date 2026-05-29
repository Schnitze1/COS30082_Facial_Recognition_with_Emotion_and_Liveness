"""
src/training/emotion_detection/train_emotion_cnn_vanilla.py

Trains Model 1, a custom CNN baseline for emotion detection, on the
folder-structured AffectNet dataset. The pipeline loads 96x96 RGB images
from local Train/Test folders, applies moderate augmentation, saves the best
model checkpoint, and evaluates performance on the held-out test set.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)
tf.random.set_seed(42)


# Configuration
class Config:
    """Centralises dataset paths, hyperparameters, and output locations.

    Keeping these values together makes the training run reproducible and
    reduces the chance of evaluating or exporting a model with mismatched paths.
    """

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR = os.environ.get(
        "AFFECTNET_DATA_DIR",
        os.path.join(PROJECT_ROOT, "data", "emotion_detection", "AffecNet")
    )
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "Train")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "Test")

    IMG_SIZE = (96, 96)
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 3e-4
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 12
    USE_CLASS_WEIGHTS = True
    MODEL_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection", "vanilla", "emotion_cnn_vanilla.keras"
    )
    H5_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection", "vanilla", "emotion_cnn_vanilla.h5"
    )
    HISTORY_SAVE_PATH = os.path.join(
        PROJECT_ROOT,
        "models",
        "emotion_detection",
        "vanilla",
        "emotion_cnn_vanilla_history.json",
    )

    # Keep this order fixed so generator labels, output neurons, evaluation, and
    # webcam prediction mapping all refer to the same emotion index.
    CLASSES = [
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]

    @classmethod
    def print_startup_info(cls):
        """Print resolved paths so local dataset mistakes are visible before training."""
        print(f"PROJECT_ROOT: {cls.PROJECT_ROOT}")
        print(f"DATA_DIR: {cls.DATA_DIR}")
        print(f"TRAIN_IMG_DIR: {cls.TRAIN_IMG_DIR}")
        print(f"TEST_IMG_DIR: {cls.TEST_IMG_DIR}")
        print(f"Train exists: {os.path.isdir(cls.TRAIN_IMG_DIR)}")
        print(f"Test exists: {os.path.isdir(cls.TEST_IMG_DIR)}")


# Dataset Loading and Generators
class EmotionDataLoader:
    """Builds the folder-based AffectNet training pipeline.

    The CSV file is intentionally not used here because the cleaned workflow
    relies on class folders as the source of truth for labels.
    """

    def __init__(self, config: Config):
        self.config = config
        self._load_datasets()
        self._create_generators()

    def _load_datasets(self):
        """Build train and test dataframes directly from class folders."""
        self._validate_dataset_folders()
        self.df = self._scan_image_folder(self.config.TRAIN_IMG_DIR)
        self.test_df = self._scan_image_folder(self.config.TEST_IMG_DIR)
        self._print_dataset_summary("Scanned train", self.df)
        self._print_dataset_summary("Scanned test", self.test_df)

    def _validate_dataset_folders(self):
        """Validate required folders before any long-running training starts."""
        for folder_name, folder_path in (
            ("Train", self.config.TRAIN_IMG_DIR),
            ("Test", self.config.TEST_IMG_DIR)
        ):
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"{folder_name} folder not found at {folder_path}. "
                    "Set AFFECTNET_DATA_DIR to the folder containing Train and Test."
                )

            existing_classes = {
                class_name.lower().strip()
                for class_name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, class_name))
            }
            missing_classes = [
                class_name
                for class_name in self.config.CLASSES
                if class_name not in existing_classes
            ]
            if missing_classes:
                raise FileNotFoundError(
                    f"{folder_name} is missing expected class folder(s): "
                    f"{', '.join(missing_classes)}. Checked: {folder_path}"
                )

    def _scan_image_folder(self, root_dir):
        """Scan class folders and normalise folder names into emotion labels."""
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

        image_extensions = {".jpg", ".jpeg", ".png"}
        rows = []

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            emotion_label = class_name.lower().strip()
            if emotion_label not in self.config.CLASSES:
                print(f"Skipping unknown class folder: {class_dir}")
                continue

            for file_name in sorted(os.listdir(class_dir)):
                _, extension = os.path.splitext(file_name)
                if extension.lower() not in image_extensions:
                    continue

                rows.append({
                    "image_path": os.path.join(class_dir, file_name),
                    "emotion_label": emotion_label
                })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"No images found in dataset folder: {root_dir}")

        print(f"Total images found in {root_dir}: {len(df)}")
        return df

    def _create_generators(self):
        """Create train, validation, and test image generators."""
        # Stratification keeps minority emotions represented in validation, so
        # validation metrics are not dominated by the larger classes.
        train_df, val_df = train_test_split(
            self.df,
            test_size=self.config.VALIDATION_SPLIT,
            stratify=self.df["emotion_label"],
            random_state=42
        )
        self.class_weights = self._compute_class_weights(train_df)
        self._print_dataset_summary("Train", train_df)
        self._print_dataset_summary("Validation", val_df)
        self._print_dataset_summary("Test", self.test_df)

        # Augmentation is intentionally moderate because large geometric changes
        # can distort facial expression cues that are important for emotion labels.
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=8,
            width_shift_range=0.08,
            height_shift_range=0.08,
            horizontal_flip=True,
            zoom_range=0.08,
            brightness_range=[0.9, 1.1],
            fill_mode="nearest"
        )

        # Validation and test images are only rescaled, not augmented, so reported
        # performance reflects the model rather than random test-time transforms.
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # The explicit class list prevents Keras from using alphabetical or
        # platform-dependent folder order for output labels.
        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="image_path",
            y_col="emotion_label",
            target_size=self.config.IMG_SIZE,
            color_mode="rgb",
            batch_size=self.config.BATCH_SIZE,
            class_mode="categorical",
            classes=self.config.CLASSES,
            shuffle=True
        )

        self.val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col="image_path",
            y_col="emotion_label",
            target_size=self.config.IMG_SIZE,
            color_mode="rgb",
            batch_size=self.config.BATCH_SIZE,
            class_mode="categorical",
            classes=self.config.CLASSES,
            shuffle=False
        )

        self.test_generator = test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            x_col="image_path",
            y_col="emotion_label",
            target_size=self.config.IMG_SIZE,
            color_mode="rgb",
            batch_size=self.config.BATCH_SIZE,
            class_mode="categorical",
            classes=self.config.CLASSES,
            shuffle=False
        )

    def get_train_data(self):
        return self.train_generator

    def get_val_data(self):
        return self.val_generator

    def get_test_data(self):
        return self.test_generator

    def get_class_weights(self):
        return self.class_weights

    def _compute_class_weights(self, train_df):
        """Compute balanced class weights from class frequency in the train split."""
        classes = self.config.CLASSES
        # Balanced weights are data-driven, not arbitrary:
        # total_samples / (number_of_classes * class_count).
        # Underrepresented classes receive higher weights, while
        # overrepresented classes receive lower weights.
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(classes),
            y=train_df["emotion_label"].to_numpy()
        )
        return {class_index: float(weight) for class_index, weight in enumerate(weights)}

    def _print_dataset_summary(self, name, df):
        print(f"{name} dataframe size: {len(df)}")
        print(f"{name} class distribution:")
        print(df["emotion_label"].value_counts().reindex(self.config.CLASSES, fill_value=0))

    def print_generator_sanity_check(self):
        """Confirm image shape, label shape, and pixel scale before fitting."""
        image_batch, label_batch = next(self.train_generator)
        print(f"Train generator class indices: {self.train_generator.class_indices}")
        print(f"Image batch shape: {image_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")
        print(f"Min pixel value: {image_batch.min():.4f}")
        print(f"Max pixel value: {image_batch.max():.4f}")


# Vanilla CNN Architecture
class CustomEmotionCNN:
    """Vanilla custom CNN trained from scratch for eight emotion classes."""

    def __init__(self, config: Config):
        self.config = config
        self.model = self._build()

    def _build(self):
        """Construct the convolutional baseline used as Model 1."""
        inputs = layers.Input(shape=(self.config.IMG_SIZE[0],
                                     self.config.IMG_SIZE[1],
                                     3))

        # Each block doubles feature capacity while pooling gradually reduces
        # spatial resolution, which is suitable for compact 96x96 face crops.
        x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.15)(x)

        # BatchNorm stabilises optimisation, while Dropout reduces reliance on
        # a small set of expression-specific activations.
        x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.20)(x)

        # Block 3
        x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Block 4
        x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.30)(x)

        x = layers.GlobalAveragePooling2D()(x)  # Avoids a large dense layer after convolutional feature extraction.
        x = layers.Dense(256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.50)(x)
        outputs = layers.Dense(len(self.config.CLASSES),
                               activation="softmax")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="CustomEmotionCNN")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                # Top-2 accuracy is useful for emotion recognition because several
                # facial expressions can be visually ambiguous.
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")
            ]
        )
        return model

    def get_model(self):
        return self.model


# Training, Evaluation, and Export
class EmotionTrainer:
    """Handles fitting, validation checkpointing, evaluation, and history export."""

    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.history = None

    def train(self, train_gen, val_gen, class_weights):
        """Train the model while keeping the best validation checkpoint."""
        os.makedirs(os.path.dirname(self.config.MODEL_SAVE_PATH), exist_ok=True)
        fit_kwargs = {}
        if self.config.USE_CLASS_WEIGHTS:
            fit_kwargs["class_weight"] = class_weights

        callbacks_list = [
            # Early stopping avoids continuing once validation loss stops improving,
            # while restore_best_weights prevents the final epoch from replacing the best model.
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=self.config.MODEL_SAVE_PATH,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config.EPOCHS,
            callbacks=callbacks_list,
            verbose=1,
            **fit_kwargs
        )

    def save_model(self):
        """Save the trained model in native Keras and H5 compatibility formats."""
        os.makedirs(os.path.dirname(self.config.MODEL_SAVE_PATH), exist_ok=True)
        self.model.save(self.config.MODEL_SAVE_PATH)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")
        try:
            self.model.save(self.config.H5_SAVE_PATH)
            print(f"H5 model saved to {self.config.H5_SAVE_PATH}")
        except Exception as exc:
            print(f"Could not save H5 model: {exc}")

    def evaluate(self, test_gen):
        """Evaluate the trained model on the held-out test set."""
        results = self.model.evaluate(test_gen, verbose=1)

        test_loss = results[0]
        test_accuracy = results[1]
        test_top_2_accuracy = results[2]

        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test top-2 accuracy: {test_top_2_accuracy:.4f}")

        return test_loss, test_accuracy, test_top_2_accuracy

    def plot_history(self):
        """Print the best validation accuracy reached during training."""
        if self.history:
            best_acc = max(self.history.history["val_accuracy"])
            print(f"Best validation accuracy: {best_acc:.4f}")

    def save_history(self):
        """Save training curves so reports can compare learning behaviour later."""
        if not self.history:
            print("No training history available to save.")
            return

        os.makedirs(os.path.dirname(self.config.HISTORY_SAVE_PATH), exist_ok=True)
        history_data = {
            metric_name: [float(value) for value in metric_values]
            for metric_name, metric_values in self.history.history.items()
            if metric_name in {
                "loss",
                "accuracy",
                "top_2_accuracy",
                "val_loss",
                "val_accuracy",
                "val_top_2_accuracy"
            }
        }

        with open(self.config.HISTORY_SAVE_PATH, "w", encoding="utf-8") as history_file:
            json.dump(history_data, history_file, indent=4)

        print(f"Training history saved to {self.config.HISTORY_SAVE_PATH}")


# Script Entry Point
if __name__ == "__main__":
    # Configuration and path diagnostics
    config = Config()
    config.print_startup_info()

    # Dataset loading
    print("Loading data...")
    data_loader = EmotionDataLoader(config)
    train_gen = data_loader.get_train_data()
    val_gen = data_loader.get_val_data()
    test_gen = data_loader.get_test_data()
    class_weights = data_loader.get_class_weights()
    data_loader.print_generator_sanity_check()

    # Model construction
    print("Building model...")
    emotion_cnn = CustomEmotionCNN(config)
    model = emotion_cnn.get_model()
    model.summary()

    # Training
    print("Starting training...")
    trainer = EmotionTrainer(model, config)
    trainer.train(train_gen, val_gen, class_weights)

    # Model and history export
    trainer.save_model()
    trainer.plot_history()
    trainer.save_history()

    # Held-out test evaluation
    print("Evaluating on test data...")
    trainer.evaluate(test_gen)

    print("Training complete.")



"""
EfficientNet-B0 Emotion Detection Training Pipeline

Transfer learning approach using EfficientNet-B0 backbone pre-trained on ImageNet.
Fine-tunes the top layers on 96x96 AffectNet emotion classification task.
Includes advanced augmentation, class balancing, and mixed precision training.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)
tf.random.set_seed(42)


class Config:
    """EfficientNet training configuration."""

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.environ.get(
        "AFFECTNET_DATA_DIR",
        os.path.join(PROJECT_ROOT, "data", "AffectNet")
    )
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "Train")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "Test")

    IMG_SIZE = (96, 96)
    BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 32))
    EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 80))
    LEARNING_RATE = float(os.environ.get("TRAIN_LR", 2e-4))
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10

    MODEL_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection_2", "efficientnet", "emotion_efficientnet.keras"
    )
    H5_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection_2", "efficientnet", "emotion_efficientnet.h5"
    )
    HISTORY_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection_2", "efficientnet", "emotion_efficientnet_history.json"
    )

    CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    USE_CLASS_WEIGHTS = True

    @classmethod
    def print_startup_info(cls):
        print(f"PROJECT_ROOT: {cls.PROJECT_ROOT}")
        print(f"DATA_DIR: {cls.DATA_DIR}")
        print(f"TRAIN_IMG_DIR: {cls.TRAIN_IMG_DIR}")
        print(f"TEST_IMG_DIR: {cls.TEST_IMG_DIR}")
        print(f"Train exists: {os.path.isdir(cls.TRAIN_IMG_DIR)}")
        print(f"Test exists: {os.path.isdir(cls.TEST_IMG_DIR)}")


class EmotionDataLoader:
    """Builds the AffectNet training pipeline with aggressive augmentation."""

    def __init__(self, config: Config):
        self.config = config
        self._load_datasets()
        self._create_generators()

    def _load_datasets(self):
        self._validate_dataset_folders()
        self.df = self._scan_image_folder(self.config.TRAIN_IMG_DIR)
        self.test_df = self._scan_image_folder(self.config.TEST_IMG_DIR)
        self._print_dataset_summary("Scanned train", self.df)
        self._print_dataset_summary("Scanned test", self.test_df)

    def _validate_dataset_folders(self):
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
                class_name for class_name in self.config.CLASSES
                if class_name not in existing_classes
            ]
            if missing_classes:
                raise FileNotFoundError(
                    f"{folder_name} is missing: {', '.join(missing_classes)}"
                )

    def _scan_image_folder(self, root_dir):
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
            raise ValueError(f"No images found in: {root_dir}")
        print(f"Total images found in {root_dir}: {len(df)}")
        return df

    def _create_generators(self):
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

        # Aggressive augmentation for transfer learning regularization
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.12,
            height_shift_range=0.12,
            horizontal_flip=True,
            zoom_range=0.12,
            brightness_range=[0.85, 1.15],
            channel_shift_range=15,
            fill_mode="nearest"
        )
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

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

    def _compute_class_weights(self, df):
        class_counts = df["emotion_label"].value_counts()
        weights = compute_class_weight(
            "balanced",
            classes=np.array(self.config.CLASSES),
            y=df["emotion_label"]
        )
        return dict(zip(self.config.CLASSES, weights))

    def _print_dataset_summary(self, label, df):
        print(f"\n{label} Dataset:")
        print(f"  Total samples: {len(df)}")
        print(f"  Class distribution:\n{df['emotion_label'].value_counts()}")

    def get_train_data(self):
        return self.train_generator

    def get_val_data(self):
        return self.val_generator

    def get_test_data(self):
        return self.test_generator

    def get_class_weights(self):
        return self.class_weights

    def get_steps_per_epoch(self):
        return len(self.train_generator)


class EfficientNetEmotionModel:
    """Transfer learning with EfficientNet-B0 backbone."""

    def __init__(self, config: Config):
        self.config = config
        self.model = self._build()

    def _build(self):
        """Build EfficientNet-B0 with custom emotion head."""
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*self.config.IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        inputs = layers.Input(shape=(*self.config.IMG_SIZE, 3))
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.50)(x)
        outputs = layers.Dense(len(self.config.CLASSES), activation="softmax")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetEmotion")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")]
        )
        return model

    def unfreeze_backbone(self, num_layers: int = 20):
        """Unfreeze top N layers of EfficientNet for fine-tuning."""
        base_model = self.model.layers[2]
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False

    def get_model(self):
        return self.model


class EmotionTrainer:
    """Handles training, validation, and export."""

    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.history = None

    def train(self, train_gen, val_gen, class_weights, steps_per_epoch):
        """Two-phase training: frozen backbone, then fine-tuning."""
        os.makedirs(os.path.dirname(self.config.MODEL_SAVE_PATH), exist_ok=True)

        print("\n[Phase 1] Training with frozen backbone...")
        phase1_callbacks = self._get_callbacks(patience=8)
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.EPOCHS // 2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=phase1_callbacks,
            verbose=1
        )

        print("\n[Phase 2] Fine-tuning with unfrozen backbone...")
        self.model.unfreeze_backbone(num_layers=20)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")]
        )
        phase2_callbacks = self._get_callbacks(patience=10)
        history2 = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.EPOCHS // 2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=phase2_callbacks,
            verbose=1
        )

        # Merge histories
        for key in history2.history:
            self.history.history[key].extend(history2.history[key])

    def _get_callbacks(self, patience: int = 10):
        return [
            callbacks.ModelCheckpoint(
                self.config.MODEL_SAVE_PATH,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
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

    def evaluate(self, test_gen):
        """Evaluate on test set."""
        test_loss, test_acc, test_top2 = self.model.evaluate(test_gen, verbose=1)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-2 Accuracy: {test_top2:.4f}")
        return {"loss": test_loss, "accuracy": test_acc, "top_2_accuracy": test_top2}

    def export_history(self):
        """Save training history to JSON."""
        os.makedirs(os.path.dirname(self.config.HISTORY_SAVE_PATH), exist_ok=True)
        with open(self.config.HISTORY_SAVE_PATH, "w") as f:
            json.dump(self.history.history, f, indent=2)
        print(f"History saved to: {self.config.HISTORY_SAVE_PATH}")

    def export_h5(self):
        """Export model to H5 format."""
        self.model.save(self.config.H5_SAVE_PATH)
        print(f"Model exported to: {self.config.H5_SAVE_PATH}")


def main():
    config = Config()
    config.print_startup_info()

    print("\n[1/3] Loading datasets...")
    data_loader = EmotionDataLoader(config)

    print("\n[2/3] Building EfficientNet model...")
    model_builder = EfficientNetEmotionModel(config)
    model = model_builder.get_model()
    model.summary()

    print("\n[3/3] Training model...")
    trainer = EmotionTrainer(model, config)
    trainer.train(
        data_loader.get_train_data(),
        data_loader.get_val_data(),
        data_loader.get_class_weights(),
        data_loader.get_steps_per_epoch()
    )

    print("\n[EVAL] Evaluating on test set...")
    trainer.evaluate(data_loader.get_test_data())

    print("\n[EXPORT] Saving model and history...")
    trainer.export_history()
    trainer.export_h5()

    print("\n✓ EfficientNet training complete!")
    print(f"  Model: {config.MODEL_SAVE_PATH}")
    print(f"  History: {config.HISTORY_SAVE_PATH}")


if __name__ == "__main__":
    main()

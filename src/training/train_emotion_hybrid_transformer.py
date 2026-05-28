"""
Hybrid CNN-Transformer Emotion Detection Training Pipeline

Combines CNN stem for initial feature extraction with Vision Transformer blocks
for global attention over patches. Includes custom learning rate schedule
(WarmupCosineDecay) and advanced augmentation for 96x96 AffectNet task.
"""

import os
import json
import math
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
    """Hybrid Transformer training configuration."""

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.environ.get(
        "AFFECTNET_DATA_DIR",
        os.path.join(PROJECT_ROOT, "data", "AffectNet")
    )
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "Train")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "Test")

    IMG_SIZE = (96, 96)
    PATCH_SIZE = 8
    BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 32))
    EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 100))
    LEARNING_RATE = float(os.environ.get("TRAIN_LR", 5e-4))
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 12

    MODEL_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection_2", "hybrid_transformer", "emotion_hybrid_transformer.keras"
    )
    H5_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection_2", "hybrid_transformer", "emotion_hybrid_transformer.h5"
    )
    HISTORY_SAVE_PATH = os.path.join(
        PROJECT_ROOT, "models", "emotion_detection_2", "hybrid_transformer", "emotion_hybrid_transformer_history.json"
    )

    CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    USE_CLASS_WEIGHTS = True

    # Transformer hyperparameters
    NUM_PATCHES = (IMG_SIZE[0] // PATCH_SIZE) ** 2
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    FF_DIM = 512
    NUM_TRANSFORMER_BLOCKS = 4
    DROPOUT_RATE = 0.15

    @classmethod
    def print_startup_info(cls):
        print(f"PROJECT_ROOT: {cls.PROJECT_ROOT}")
        print(f"DATA_DIR: {cls.DATA_DIR}")
        print(f"IMG_SIZE: {cls.IMG_SIZE}, PATCH_SIZE: {cls.PATCH_SIZE}")
        print(f"NUM_PATCHES: {cls.NUM_PATCHES}")
        print(f"EMBEDDING_DIM: {cls.EMBEDDING_DIM}, NUM_HEADS: {cls.NUM_HEADS}")
        print(f"Train exists: {os.path.isdir(cls.TRAIN_IMG_DIR)}")
        print(f"Test exists: {os.path.isdir(cls.TEST_IMG_DIR)}")


class EmotionDataLoader:
    """Builds the AffectNet training pipeline."""

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

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=12,
            width_shift_range=0.10,
            height_shift_range=0.10,
            horizontal_flip=True,
            zoom_range=0.10,
            brightness_range=[0.90, 1.10],
            channel_shift_range=10,
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


class PatchPositionEmbedding(layers.Layer):
    """Learnable position embeddings for patches."""

    def __init__(self, num_patches: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.pos_embed = layers.Embedding(num_patches, embedding_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_embed(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "embedding_dim": self.embedding_dim})
        return cfg


class TransformerBlock(layers.Layer):
    """Multi-head self-attention + FFN with residual connections."""

    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embedding_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        normed = self.norm1(x)
        attn = self.attention(normed, normed, training=training)
        attn = self.drop1(attn, training=training)
        x = x + attn
        normed = self.norm2(x)
        ff = self.ffn(normed)
        ff = self.drop2(ff, training=training)
        return x + ff

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup then cosine decay."""

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
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


class HybridTransformerEmotionModel:
    """Hybrid CNN-Transformer architecture."""

    def __init__(self, config: Config):
        self.config = config
        self.model = self._build()

    def _build(self):
        """Build CNN stem + Vision Transformer."""
        cfg = self.config
        inputs = layers.Input(shape=(*cfg.IMG_SIZE, 3))

        # CNN stem: extract initial features
        x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Patch embedding: project spatial features to embedding dimension
        x = layers.Conv2D(cfg.EMBEDDING_DIM, cfg.PATCH_SIZE, strides=cfg.PATCH_SIZE, padding="valid")(x)
        x = layers.Reshape((cfg.NUM_PATCHES, cfg.EMBEDDING_DIM))(x)

        # Position embeddings
        x = PatchPositionEmbedding(cfg.NUM_PATCHES, cfg.EMBEDDING_DIM)(x)

        # Transformer blocks
        for i in range(cfg.NUM_TRANSFORMER_BLOCKS):
            x = TransformerBlock(
                cfg.EMBEDDING_DIM,
                cfg.NUM_HEADS,
                cfg.FF_DIM,
                cfg.DROPOUT_RATE,
                name=f"transformer_block_{i}"
            )(x)

        # Global average pooling + classification head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.50)(x)
        outputs = layers.Dense(len(cfg.CLASSES), activation="softmax")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="HybridTransformerEmotion")
        return model

    def compile_with_schedule(self, total_steps: int):
        """Compile with WarmupCosineDecay schedule."""
        lr_schedule = WarmupCosineDecay(
            base_lr=self.config.LEARNING_RATE,
            warmup_steps=int(0.1 * total_steps),
            total_steps=total_steps,
            min_lr=1e-6
        )
        optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")]
        )

    def get_model(self):
        return self.model


class EmotionTrainer:
    """Handles training, validation, and export."""

    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.history = None

    def train(self, train_gen, val_gen, class_weights, steps_per_epoch):
        """Train with custom learning rate schedule."""
        os.makedirs(os.path.dirname(self.config.MODEL_SAVE_PATH), exist_ok=True)

        total_steps = steps_per_epoch * self.config.EPOCHS
        self.model.compile_with_schedule(total_steps)
        self.model.get_model().summary()

        callbacks_list = [
            callbacks.ModelCheckpoint(
                self.config.MODEL_SAVE_PATH,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
        ]

        self.history = self.model.get_model().fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.EPOCHS,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )

    def evaluate(self, test_gen):
        """Evaluate on test set."""
        test_loss, test_acc, test_top2 = self.model.get_model().evaluate(test_gen, verbose=1)
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
        self.model.get_model().save(self.config.H5_SAVE_PATH)
        print(f"Model exported to: {self.config.H5_SAVE_PATH}")


def main():
    config = Config()
    config.print_startup_info()

    print("\n[1/3] Loading datasets...")
    data_loader = EmotionDataLoader(config)

    print("\n[2/3] Building Hybrid Transformer model...")
    model_builder = HybridTransformerEmotionModel(config)
    model = model_builder.get_model()

    print("\n[3/3] Training model...")
    trainer = EmotionTrainer(model_builder, config)
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

    print("\n✓ Hybrid Transformer training complete!")
    print(f"  Model: {config.MODEL_SAVE_PATH}")
    print(f"  History: {config.HISTORY_SAVE_PATH}")


if __name__ == "__main__":
    main()

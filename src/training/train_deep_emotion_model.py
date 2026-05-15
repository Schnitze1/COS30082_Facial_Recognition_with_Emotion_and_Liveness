"""
src/training/train_deep_emotion_model.py

Trains Model 2, a deeper custom CNN for emotion detection, on the
folder-structured AffectNet dataset. This model is trained from scratch and
uses residual learning with squeeze-and-excitation channel attention.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import mixed_precision

from train_emotion_model import Config, EmotionDataLoader, EmotionTrainer

np.random.seed(42)
tf.random.set_seed(42)


class DeepConfig(Config):
    """Configuration for Model 2 outputs and optimisation settings."""

    MODEL_SAVE_PATH = os.path.join(Config.PROJECT_ROOT, "models", "emotion_deep_cnn.keras")
    HISTORY_SAVE_PATH = os.path.join(Config.PROJECT_ROOT, "models", "emotion_deep_cnn_history.json")
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 12
    USE_CLASS_WEIGHTS = True


def print_hardware_info():
    """Print TensorFlow hardware diagnostics and configure GPU memory growth."""
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs detected:", gpus)

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print("Could not set memory growth:", e)
    else:
        print("No GPU detected by TensorFlow. Training will run on CPU.")


def configure_mixed_precision():
    """Enable mixed precision only when TensorFlow detects a GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")
    else:
        print("Mixed precision not enabled because no GPU was detected.")


def se_block(input_tensor, reduction=16):
    """Apply squeeze-and-excitation channel attention."""
    channels = input_tensor.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    excitation = layers.Dense(max(channels // reduction, 8), activation="relu")(squeeze)
    excitation = layers.Dense(channels, activation="sigmoid")(excitation)
    excitation = layers.Reshape((1, 1, channels))(excitation)
    return layers.Multiply()([input_tensor, excitation])


def residual_se_block(x, filters, dropout_rate, block_name):
    """Build a residual convolution block with squeeze-and-excitation attention."""
    shortcut = x

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            padding="same",
            use_bias=False,
            name=f"{block_name}_shortcut_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{block_name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{block_name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{block_name}_out_relu")(x)
    x = se_block(x, reduction=16)
    x = layers.MaxPooling2D((2, 2), name=f"{block_name}_pool")(x)
    x = layers.Dropout(dropout_rate, name=f"{block_name}_dropout")(x)

    return x


class DeepEmotionCNN:
    """Deep custom CNN with residual blocks and squeeze-and-excitation attention."""

    def __init__(self, config: DeepConfig):
        self.config = config
        self.model = self._build()

    def _build(self):
        """Construct and compile the Model 2 architecture."""
        inputs = layers.Input(shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], 3))

        x = residual_se_block(inputs, 64, 0.20, "block1")
        x = residual_se_block(x, 128, 0.25, "block2")
        x = residual_se_block(x, 256, 0.30, "block3")
        x = residual_se_block(x, 512, 0.35, "block4")

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.50)(x)
        x = layers.Dense(256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.40)(x)
        outputs = layers.Dense(
            len(self.config.CLASSES),
            activation="softmax",
            dtype="float32"
        )(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="DeepEmotionCNN")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")
            ]
        )
        return model

    def get_model(self):
        """Return the compiled Keras model."""
        return self.model


if __name__ == "__main__":
    config = DeepConfig()
    print("Running Model 2: Deep CNN with Residual and Squeeze-and-Excitation blocks")
    print("Model 2 will be saved separately and will not overwrite Model 1.")
    print_hardware_info()
    configure_mixed_precision()
    config.print_startup_info()

    print("Loading data...")
    data_loader = EmotionDataLoader(config)
    train_gen = data_loader.get_train_data()
    val_gen = data_loader.get_val_data()
    test_gen = data_loader.get_test_data()
    class_weights = data_loader.get_class_weights()
    data_loader.print_generator_sanity_check()

    print("Building model...")
    deep_cnn = DeepEmotionCNN(config)
    model = deep_cnn.get_model()
    model.summary()

    print("Starting training...")
    trainer = EmotionTrainer(model, config)
    trainer.train(train_gen, val_gen, class_weights)

    trainer.save_model()
    trainer.plot_history()
    trainer.save_history()

    print("Evaluating on test data...")
    trainer.evaluate(test_gen)

    print("Training complete.")

"""
Trains Model 2 v2: An improved bottleneck Residual SE CNN for emotion detection.
This version is trained 100% from scratch on AffectNet, with major architectural
refinements for stability and accuracy. Saves separately from Model 1 & Model 2.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    AdamW = None

from train_emotion_model import Config, EmotionDataLoader, EmotionTrainer

np.random.seed(42)
tf.random.set_seed(42)


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
class DeepV2Config(Config):
    """Configuration for the improved Model 2 v2 experiment."""

    MODEL_SAVE_PATH = os.path.join(Config.PROJECT_ROOT, "models", "emotion_deep_cnn_v2.keras")
    HISTORY_SAVE_PATH = os.path.join(Config.PROJECT_ROOT, "models", "emotion_deep_cnn_v2_history.json")

    IMG_SIZE = (112, 112)
    LEARNING_RATE = 3e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 18
    USE_CLASS_WEIGHTS = True
    USE_COSINE_ANNEALING = True


# -------------------------------------------------------------------------
# Hardware setup
# -------------------------------------------------------------------------
def print_hardware_info():
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
        print("No GPU detected — training will run on CPU.")


def configure_mixed_precision():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")
    else:
        print("Mixed precision NOT enabled (no GPU detected).")


# -------------------------------------------------------------------------
# Squeeze-and-Excitation
# -------------------------------------------------------------------------
def se_block(x, reduction=16):
    """Improved SE block."""
    channels = x.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(x)
    excitation = layers.Dense(max(channels // reduction, 8), activation="relu")(squeeze)
    excitation = layers.Dense(channels, activation="sigmoid")(excitation)
    excitation = layers.Reshape((1, 1, channels))(excitation)
    return layers.Multiply()([x, excitation])


# -------------------------------------------------------------------------
# Bottleneck Residual Block
# -------------------------------------------------------------------------
def residual_se_bottleneck(x, filters, stride=1, dropout_rate=0.0, name=None):
    shortcut = x

    # 1x1 reduce
    y = layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu")(y)

    # 3x3 conv
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = layers.Activation("relu")(y)

    # 1x1 expand
    y = layers.Conv2D(filters * 4, 1, strides=1, padding="same", use_bias=False, name=f"{name}_conv3")(y)
    y = layers.BatchNormalization(name=f"{name}_bn3")(y)

    # match shortcut
    if shortcut.shape[-1] != filters * 4 or stride != 1:
        shortcut = layers.Conv2D(filters * 4, 1, strides=stride, padding="same", use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    # add
    y = layers.Add(name=f"{name}_add")([y, shortcut])
    y = layers.Activation("relu", name=f"{name}_post_add_relu")(y)

    # SE
    y = se_block(y)

    # stability: relu → dropout
    y = layers.Activation("relu", name=f"{name}_final_relu")(y)

    if dropout_rate > 0:
        y = layers.Dropout(dropout_rate, name=f"{name}_dropout")(y)

    return y


# -------------------------------------------------------------------------
# CNN Model
# -------------------------------------------------------------------------
class DeepEmotionCNNV2:
    """Improved CNN with bottleneck SE blocks and proper classifier head."""

    def __init__(self, config: DeepV2Config):
        self.config = config
        self.model = self._build()

    # -----------------------------------------------------
    def _build(self):
        inputs = layers.Input(shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], 3))

        # Stem
        x = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

        # -------------------------
        # Stage 1 (3 blocks)
        x = residual_se_bottleneck(x, 64, stride=1, dropout_rate=0.05, name="block1_1")
        x = residual_se_bottleneck(x, 64, stride=1, name="block1_2")
        x = residual_se_bottleneck(x, 64, stride=1, name="block1_3")

        # Stage 2 (3 blocks)
        x = residual_se_bottleneck(x, 128, stride=2, dropout_rate=0.10, name="block2_1")
        x = residual_se_bottleneck(x, 128, stride=1, name="block2_2")
        x = residual_se_bottleneck(x, 128, stride=1, name="block2_3")

        # Stage 3 (3 blocks)
        x = residual_se_bottleneck(x, 256, stride=2, dropout_rate=0.15, name="block3_1")
        x = residual_se_bottleneck(x, 256, stride=1, name="block3_2")
        x = residual_se_bottleneck(x, 256, stride=1, name="block3_3")

        # Stage 4 (2 blocks)
        x = residual_se_bottleneck(x, 512, stride=2, dropout_rate=0.20, name="block4_1")
        x = residual_se_bottleneck(x, 512, stride=1, name="block4_2")

        # -----------------------------------------------------
        # Classifier Head
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

        model = models.Model(inputs=inputs, outputs=outputs, name="DeepEmotionCNNV2")
        model.compile(
            optimizer=self._build_optimizer(),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                "accuracy",
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")
            ]
        )
        return model

    # -----------------------------------------------------
    def _build_optimizer(self):
        base_lr = self.config.LEARNING_RATE

        # Use AdamW when available
        if AdamW is not None:
            opt = AdamW(learning_rate=base_lr, weight_decay=1e-5, clipnorm=1.0)
        else:
            print("AdamW unavailable, falling back to Adam.")
            opt = tf.keras.optimizers.Adam(learning_rate=base_lr, clipnorm=1.0)

        # Mixed precision safety
        if mixed_precision.global_policy().name == "mixed_float16":
            opt = mixed_precision.LossScaleOptimizer(opt)

        return opt

    def get_model(self):
        return self.model


# -------------------------------------------------------------------------
# Cosine Annealing LR
# -------------------------------------------------------------------------
def cosine_annealing_scheduler(epoch, lr_max=3e-4, lr_min=1e-6, T_max=50):
    cos_inner = (np.pi * (epoch % T_max)) / T_max
    return lr_min + (lr_max - lr_min) * (1 + np.cos(cos_inner)) / 2


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    config = DeepV2Config()

    print("Running Model 2 v2: Bottleneck Residual SE CNN (Improved)")
    print("Features: 112x112 input, deep SE bottlenecks, AdamW, BN-stabilized classifier.")

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
    model = DeepEmotionCNNV2(config).get_model()
    model.summary()

    print("Starting training...")
    trainer = EmotionTrainer(model, config)

    callbacks = []

    if config.USE_COSINE_ANNEALING:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(cosine_annealing_scheduler))

    trainer.train(train_gen, val_gen, class_weights, extra_callbacks=callbacks)

    trainer.save_model()
    trainer.plot_history()
    trainer.save_history()

    print("Evaluating on test data...")
    trainer.evaluate(test_gen)

    print("Training complete.")

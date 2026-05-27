import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

#  Paths 
BASE_DIR   = Path(__file__).resolve().parents[3]          # project root
DATA_DIR   = BASE_DIR / "data" / "anti_spoofing" / "LCC_FASD"
TRAIN_DIR  = DATA_DIR / "train"
DEV_DIR    = DATA_DIR / "dev"
TEST_DIR   = DATA_DIR / "test"
MODEL_DIR  = BASE_DIR / "models" / "anti_spoofing"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH       = MODEL_DIR / "antispoofing_model.h5"
MODEL_PATH_KERAS = MODEL_DIR / "antispoofing_model.keras"

#  Hyperparameters 
IMG_SIZE    = 128          # resize all images to 128x128
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
FINE_TUNE_AT = 100         # unfreeze MobileNetV2 layers from this index onwards

#  1. Data Generators 
# Training: augment to improve generalisation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=10,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
)

# Dev / Test: only rescale
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",       # real=1, fake=0
    classes=["fake", "real"],  # fake→0, real→1
    shuffle=True,
)

dev_gen = val_datagen.flow_from_directory(
    DEV_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["fake", "real"],
    shuffle=False,
)

test_gen = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["fake", "real"],
    shuffle=False,
)

print(f"Class indices: {train_gen.class_indices}")   # {'fake': 0, 'real': 1}
print(f"Training samples  : {train_gen.samples}")
print(f"Development samples: {dev_gen.samples}")
print(f"Test samples      : {test_gen.samples}")

#  Class Weights (handles imbalanced real/fake split) 
class_weights_array = compute_class_weight(
    "balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weights = dict(enumerate(class_weights_array))
print(f"Class weights: {class_weights}")

#  2. Build Model (Transfer Learning + Fine-Tuning) 
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze all base layers first
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)   # binary: real vs fake

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

#  3. Phase 1 — Train head only 
print("\n=== Phase 1: Training classification head ===")

callbacks = [
    ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
]

history1 = model.fit(
    train_gen,
    validation_data=dev_gen,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks,
)

#  4. Phase 2 — Fine-tune top layers of MobileNetV2 
print(f"\n=== Phase 2: Fine-tuning from layer {FINE_TUNE_AT} onwards ===")

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Lower LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR / 10),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_gen,
    validation_data=dev_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

print(f"\nBest model saved to: {MODEL_PATH}")

# Also save in .keras format
model.save(str(MODEL_PATH_KERAS))
print(f"Also saved as: {MODEL_PATH_KERAS}")

#  5. Evaluation on Test Set 
print("\n=== Evaluating on test set ===")
model.load_weights(str(MODEL_PATH))

test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss    : {test_loss:.4f}")

# Predictions for ROC / AUC
y_true  = test_gen.classes
y_pred  = model.predict(test_gen, verbose=1).flatten()
y_label = (y_pred >= 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_label, target_names=["fake", "real"]))

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc     = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

# 6. Save Plots 
report_dir = BASE_DIR / "reports" / "anti_spoofing"
report_dir.mkdir(parents=True, exist_ok=True)

# Combine both training phases
acc  = history1.history["accuracy"]  + history2.history["accuracy"]
val  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

# Accuracy plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val, label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(report_dir / "training_curves.png")
print(f"Training curves saved.")

# ROC curve plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Anti-Spoofing")
plt.legend()
plt.savefig(report_dir / "roc_curve.png")
print(f"ROC curve saved.")

# Confusion matrix
cm = confusion_matrix(y_true, y_label)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["fake", "real"], yticklabels=["fake", "real"])
plt.title("Confusion Matrix - Anti-Spoofing")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig(report_dir / "confusion_matrix.png")
print(f"Confusion matrix saved.")

# Save training history as JSON
combined_history = {
    "accuracy"    : acc,
    "val_accuracy": val,
    "loss"        : loss,
    "val_loss"    : val_loss,
    "auc"         : float(roc_auc),
    "test_accuracy": float(test_acc),
}
with open(report_dir / "training_history.json", "w") as f:
    json.dump(combined_history, f, indent=2)
print("Training history saved.")

print("\nTraining complete!")
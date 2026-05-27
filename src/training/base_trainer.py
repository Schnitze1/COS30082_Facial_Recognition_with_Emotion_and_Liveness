import os
import datetime
import json
import tensorflow as tf

class ImageModelTrainer:
    """
    Handles the common dataset loading and training loop for identity classification models.
    """
    def __init__(self, model_class, model_name: str, batch_size=32, img_size=(112, 112), epochs=40, learning_rate=0.001):
        """
        Initializes the ImageModelTrainer.
        
        :param model_class: The class of the model to instantiate (e.g. FNNFaceModel).
        :param model_name: String representing the model name for saving checkpoints.
        :param batch_size: Integer representing the batch size.
        :param img_size: Tuple representing image dimensions.
        :param epochs: Integer representing the maximum number of epochs.
        :param learning_rate: Float representing the learning rate.
        """
        self.model_class = model_class
        self.model_name = model_name
        self.batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", batch_size))
        self.img_size = img_size
        self.epochs = int(os.environ.get("TRAIN_EPOCHS", epochs))
        self.learning_rate = float(os.environ.get("TRAIN_LR", learning_rate))
        
        self.train_dir = 'data/classification_data/train_data'
        self.val_dir = 'data/classification_data/val_data'

    def _load_datasets(self):
        """
        Loads and caches the grayscale training and validation datasets.
        
        :return: A tuple of (train_ds, val_ds) tf.data.Dataset objects.
        """
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            labels='inferred',
            label_mode='int',
            color_mode='grayscale',
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.val_dir,
            labels='inferred',
            label_mode='int',
            color_mode='grayscale',
            image_size=self.img_size,
            batch_size=self.batch_size
        )
        
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)
        
        return train_ds, val_ds

    def train(self):
        """
        Executes the full training pipeline including compilation, callbacks, and fitting.
        """
        print(f"Loading datasets for {self.model_name}...")
        train_ds, val_ds = self._load_datasets()

        print(f"Initializing {self.model_name}...")
        face_model = self.model_class(input_shape=(self.img_size[0], self.img_size[1], 1), num_classes=24)
        model = face_model.model
        face_model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        os.makedirs('models/checkpoints', exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/checkpoints/{self.model_name}_best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        print("Starting training")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback]
        )
        
        os.makedirs('logs/history', exist_ok=True)
        with open(f'logs/history/{self.model_name}_history.json', 'w') as f:
            json.dump(history.history, f)
            
        print("Training complete.")

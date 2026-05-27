import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import tensorflow as tf
import json
from src.models.lip_cnn_lstm import LipCNNLSTM
from src.models.lip_cnn_gru import LipCNNGRU
from src.models.lip_conv3d import LipConv3D

class SequenceModelTrainer:
    """
    Handles loading the spatiotemporal numpy datasets and training multiple sequence models.
    """
    def __init__(self, data_dir='data/lip_sequence_data', epochs=30, batch_size=16):
        """
        Initializes the SequenceModelTrainer.
        
        :param data_dir: Directory containing the .npy dataset files.
        :param epochs: Integer representing maximum number of epochs.
        :param batch_size: Integer representing batch size.
        """
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def _load_dataset(self):
        """
        Loads the .npy dataset files and scales the image data to [0, 1].
        """
        try:
            self.X_train = np.load(os.path.join(self.data_dir, 'X_train.npy')).astype('float32') / 255.0
            self.y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
            self.X_val = np.load(os.path.join(self.data_dir, 'X_val.npy')).astype('float32') / 255.0
            self.y_val = np.load(os.path.join(self.data_dir, 'y_val.npy'))
        except FileNotFoundError:
            print("Dataset not found. Please run the dataset builder first.")
            sys.exit(1)

    def train_model(self, model_builder, model_name: str):
        """
        Trains a single sequence model and saves the best checkpoint and history.
        
        :param model_builder: Instantiated model builder object.
        :param model_name: String name for the model checkpoint.
        """
        print(f"Training {model_name}...")
        model = model_builder.build()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        checkpoint_path = f"models/checkpoints/{model_name}_best.h5"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy')
        ]
        
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks
        )
        
        os.makedirs('logs/history', exist_ok=True)
        with open(f'logs/history/{model_name}_history.json', 'w') as f:
            json.dump(history.history, f)

    def run(self):
        """
        Executes the data loading and training loop for all models.
        """
        print("Loading Lip Sequence Dataset...")
        self._load_dataset()
        
        models_to_train = [
            (LipCNNLSTM(), "LipCNNLSTM"),
            (LipCNNGRU(), "LipCNNGRU"),
            (LipConv3D(), "LipConv3D")
        ]
        
        for builder, name in models_to_train:
            self.train_model(builder, name)
            
        print("All Spatiotemporal models trained successfully.")

def main():
    """
    Entry point to train all lip reading sequence models.
    """
    trainer = SequenceModelTrainer()
    trainer.run()

if __name__ == '__main__':
    main()

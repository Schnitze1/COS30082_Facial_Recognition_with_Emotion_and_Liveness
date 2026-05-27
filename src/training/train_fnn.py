import os
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.fnn import FNNFaceModel
from src.training.base_trainer import ImageModelTrainer

def main():
    """
    Entry point to train the FNN Baseline model.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    trainer = ImageModelTrainer(model_class=FNNFaceModel, model_name="fnn")
    trainer.train()

if __name__ == '__main__':
    main()

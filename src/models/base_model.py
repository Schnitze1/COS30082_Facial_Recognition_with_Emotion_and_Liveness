from abc import ABC, abstractmethod
import tensorflow as tf

class BaseFaceModel(ABC):
    def __init__(self, input_shape=(112, 112, 3), num_classes=24):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    @abstractmethod
    def _build_model(self):
        """
        Abstract method to be implemented by child classes.
        
        :return: A compiled or uncompiled tf.keras.Model.
        """
        pass
        
    def summary(self):
        """
        Prints the Keras model summary.
        """
        self.model.summary()
        
    def save(self, filepath):
        """
        Saves the Keras model to disk.
        
        :param filepath: Path to save the model.
        """
        self.model.save(filepath)

import tensorflow as tf
from .base_model import BaseFaceModel

class MLPFaceModel(BaseFaceModel):
    def _build_model(self):
        """
        Builds the MLP model architecture.
        
        :return: A compiled or uncompiled tf.keras.Model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.2),
            
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, name='embedding_layer'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ], name="MLP_Face_Classifier")
        
        return model

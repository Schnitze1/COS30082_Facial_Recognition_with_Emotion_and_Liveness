import tensorflow as tf

class LipConv3D:
    def __init__(self, input_shape=(15, 64, 64, 1), num_classes=2):
        """
        Initializes the LipConv3D model.
        
        :param input_shape: Tuple representing the input shape (frames, height, width, channels).
        :param num_classes: Integer representing the number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        """
        Builds the Conv3D model architecture.
        
        :return: A compiled or uncompiled tf.keras.Model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            
            tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ], name="Lip_Conv3D")
        
        return model

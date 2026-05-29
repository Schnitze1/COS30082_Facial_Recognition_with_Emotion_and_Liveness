import tensorflow as tf

class LipCNNGRU:
    def __init__(self, input_shape=(15, 64, 64, 1), num_classes=2):
        """
        Initializes the LipCNNGRU model.
        
        :param input_shape: Tuple representing the input shape (frames, height, width, channels).
        :param num_classes: Integer representing the number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        """
        Builds the CNN-GRU model architecture.
        
        :return: A compiled or uncompiled tf.keras.Model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
            
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
            
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
            
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
            
            tf.keras.layers.GRU(128, return_sequences=False, unroll=True),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ], name="Lip_CNN_GRU")
        
        return model

import tensorflow as tf

from keras import layers

def create_cnn_model(action_space):
    return tf.keras.Sequential([
        layers.Input(shape=(4, 84, 84)),
        layers.Rescaling(scale=1.0 / 255),
        layers.Conv2D(32, 8, strides=4, activation="relu", data_format="channels_first"),
        layers.Conv2D(64, 4, strides=2, activation="relu", data_format="channels_first"),
        layers.Conv2D(64, 3, strides=1, activation="relu", data_format="channels_first"),
        layers.Flatten(data_format="channels_first"),
        layers.Dense(512, activation="relu"),
        layers.Dense(action_space, activation="linear", name="output_actions")
    ])

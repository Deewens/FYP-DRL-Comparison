import tensorflow as tf

from keras import layers

from transformers import TFSwinModel, SwinConfig


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


SWIN_CONFIG = SwinConfig(
    image_size=84,
    patch_size=3,
    num_channels=4,
    embed_dim=96,
    depths=[2, 3, 2],
    num_heads=[3, 3, 6],
    window_size=7,
    mlp_ratio=4.0,
    drop_path_rate=0.1,
)


def create_swin_model(action_space):
    input = layers.Input(shape=(4, 84, 84))
    x = layers.Rescaling(scale=1.0 / 255)(input)
    x = TFSwinModel(config=SWIN_CONFIG)(x)
    action = layers.Dense(action_space, activation="linear")(x.pooler_output)

    model = tf.keras.Model(inputs=input, outputs=action)

    return model

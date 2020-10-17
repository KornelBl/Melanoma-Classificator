import tensorflow.keras as keras
import tensorflow as tf


def EffNet0(config, trainable_base=False) -> keras.models.Sequential:
    conv_base = keras.applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=False,
                                                               input_shape=(256,256,3))
    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(keras.layers.GlobalMaxPooling2D(name="gap"))
    model.add(keras.layers.Dense(4096, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1, activation="sigmoid", name="fc_out"))
    conv_base.trainable = trainable_base
    return model

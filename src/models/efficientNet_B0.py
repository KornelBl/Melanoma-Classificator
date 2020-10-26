import tensorflow.keras as keras
import tensorflow as tf


def EffNet0(config, trainable_base=False) -> keras.models.Sequential:
    output_bias = keras.initializers.Constant(config['output_bias'])

    conv_base = keras.applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=False,
                                                               input_shape=config["input_shape"])
    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(keras.layers.GlobalMaxPooling2D(name="gap"))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation="sigmoid", name="fc_out", bias_initializer=output_bias))
    conv_base.trainable = trainable_base
    return model


def unfreeze_blocks(model: keras.models.Sequential, number_of_blocks) -> keras.models.Sequential:
    model_base = model.layers[0]
    if number_of_blocks == "all":
        model_base.trainable = True
        return model_base
    numbers = [i for i in range(7, 7-number_of_blocks, -1)]
    block_names = []
    for number in numbers:
        block_names.append(f"block{number}")
    if block_names:
        block_names.append("top_")
    for layer in model_base.layers:
        for block_name in block_names:
            print(f"{block_name}, {layer.name}")
            if block_name in layer.name:
                print(f"IT IS ON")
                layer.trainable = True
                break
    return model_base

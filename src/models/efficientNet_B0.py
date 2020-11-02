import tensorflow.keras as keras
import tensorflow as tf

def EffNet0(config, trainable_base=False) -> keras.models.Model:
    output_bias = keras.initializers.Constant(config['output_bias'])
    pd_dropout2 = None

    if config["use_patient_data"]:
        pd_input = keras.layers.Input((9,))
        pd_dense1 = keras.layers.Dense(256)(pd_input)
        pd_batch_norm1 = keras.layers.BatchNormalization()(pd_dense1)
        pd_activation1 = keras.layers.Activation('relu')(pd_batch_norm1)
        pd_dropout1 = keras.layers.Dropout(0.4)(pd_activation1)
        pd_dense2 = keras.layers.Dense(256)(pd_dropout1)
        pd_batch_norm2 = keras.layers.BatchNormalization()(pd_dense2)
        pd_activation2 = keras.layers.Activation('relu')(pd_batch_norm2)
        pd_dropout2 = keras.layers.Dropout(0.4)(pd_activation2)


    input_img = keras.layers.Input(config["input_shape"])
    conv_base = keras.applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=False,
                                                               input_shape=config["input_shape"])(input_img)
    gmp = keras.layers.GlobalMaxPooling2D(name="gap")(conv_base)
    concat = keras.layers.concatenate([pd_dropout2, gmp])
    dense1 = keras.layers.Dense(1024)(concat)
    batch_norm1 = keras.layers.BatchNormalization()(dense1)
    activation1 = keras.layers.Activation('swish')(batch_norm1)
    dropout1 = keras.layers.Dropout(0.4)(activation1)
    fc_out = keras.layers.Dense(1, activation="sigmoid", name="fc_out", bias_initializer=output_bias)(dropout1)

    model = keras.models.Model(inputs=[input_img, pd_input], outputs=fc_out)

    model.layers[1].trainable = config["trainable_base"]
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

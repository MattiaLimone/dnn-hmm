import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, TimeDistributed, Layer, InputLayer, BatchNormalization, Flatten, Dropout, Reshape, Add
from keras.utils.vis_utils import plot_model
from typing import final, Optional, Union, Any, Iterable
import tensorflow as tf

def Autoencoder(input_layer: Layer, encoder_layers: list[Layer], decoder_layers: list[Layer], bottleneck: Layer,
                jump_list: list[tuple]) -> keras.Model:

    encoder = Encoder(input_layer=input_layer, encoder_layers=encoder_layers, bottleneck=bottleneck)
    decoder = Decoder(encoder_block=encoder, decoder_layers=decoder_layers, jump_list=jump_list)

    model = Model(input_layer, decoder)

    if jump_list:
        print('jumper')

    print(model.summary())
    plot_model(model, to_file='plots/autoencoder.png', show_shapes=True, show_layer_names=True)
    return model

def Encoder(input_layer: Layer, encoder_layers: list[Layer], bottleneck: Layer)-> list:
    encoder_block = []
    for i in range(0, len(encoder_layers)):
        if i == 0:
            encoder_block.append(input_layer)
            encoder_block.append(encoder_layers[i](input_layer))
            print(f"Adding connection from Input Layer to Encoder Layer #{i}")
        else:
            encoder_block.append(encoder_layers[i](encoder_block[-1]))
            print(f"Adding connection from Encoder Layer #{i-1} to Encoder Layer #{i}")

    encoder_block.append(bottleneck(encoder_block[-1]))
    print(f"Adding connection from Encoder Layer #{i} to Bottleneck Layer")

    return encoder_block


def Decoder(encoder_block: list[Layer], decoder_layers: list[Layer], jump_list: bool) -> list:
    decoder_block = []
    counter = len(encoder_block)-2
    if jump_list:
        for i in range(0, len(decoder_layers)):
            if i == 0:
                x = decoder_layers[i](encoder_block[-1])
                decoder_block.append(x)
                print(f"Adding connection from Encoder Bottleneck to Decoder Layer #{i}")
            else:
                x = decoder_layers[i](decoder_block[-1])
                print(f"Adding connection from Decoder Layer #{i-1} to Decoder Layer #{i}")
                decoder_block.append(x)

            decoder_block.append(Add()([encoder_block[counter], x]))
            print(f"Adding jump connection from Encoder Layer #{counter} to Decoder Layer #{i}")
            counter -= 1
    else:
        for i in range(0, len(decoder_layers)):
            if i == 0:
                decoder_block.append(decoder_layers[i](encoder_block[-1]))
            else:
                decoder_block.append(decoder_layers[i](decoder_block[i - 1]))

    return decoder_block

input_img = keras.Input(shape=(784,))
#encoder list
encoded_1 = Dense(784, activation='relu', name="Livello_1_Encoder")
encoded_2 = Dense(512, activation='relu', name="Livello_2_Encoder")
encoded_3 = Dense(256, activation='relu', name="Livello_3_Encoder")

encoder = [encoded_1, encoded_2, encoded_3]
#decoder list
decoded_1 = Dense(256, activation='sigmoid', name="Livello_1_Decoder")
decoded_2 = Dense(512, activation='sigmoid', name="Livello_2_Decoder")
decoded_3 = Dense(784, activation='sigmoid', name="Livello_3_Decoder")

decoder = [decoded_1, decoded_2, decoded_3]

bottleneck = Dense(128, activation='relu')
jump_list = True
model = Autoencoder(input_layer=input_img, encoder_layers=encoder, decoder_layers=decoder, bottleneck=bottleneck,
                    jump_list=jump_list)
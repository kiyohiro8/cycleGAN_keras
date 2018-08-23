# -*- coding: utf-8 -*-

import os

from keras.layers import Input, Dense, Activation, Flatten, Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.engine.network import Network
from keras.initializers import TruncatedNormal



def mapping_function(input_shape, base_name):
    initializer = TruncatedNormal(mean=0, stddev=0.2, seed=42)
    x = in_x = Input(shape=input_shape)

    # size→size//2→size//4→size//8
    x = Conv2D(64, kernel_size=7, strides=1, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv1")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(x)
    x = Activation("relu")(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv2")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(x)
    x = Activation("relu")(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer,
               use_bias=False,
               name=base_name + "_conv3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(x)
    x = Activation("relu")(x)

    for i in range(9):
        x = residual_block(x, base_name=base_name, block_num=i, initializer=initializer)

    # size//8→size//4→size//2→size
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv2")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn6")(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer,
                        name=base_name + "_deconv3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn7")(x)
    x = Activation("relu")(x)
    out = Conv2DTranspose(3, kernel_size=7, strides=1, padding='same', activation="tanh",
                          kernel_initializer=initializer, name=base_name + "_out")(x)
    network = Network(in_x, out, name=base_name)
    return network


def discriminator(input_shape, base_name):
    initializer_d = TruncatedNormal(mean=0, stddev=0.2, seed=42)

    D = in_D = Input(shape=input_shape)
    D = Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv1")(D)
    D = LeakyReLU(0.2)(D)
    D = Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv2")(D)
    D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn1")(D)
    D = LeakyReLU(0.2)(D)
    D = Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv3")(D)
    D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn2")(D)
    D = LeakyReLU(0.2)(D)
    D = Conv2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer_d,
               use_bias=False,
               name=base_name + "_conv4")(D)
    D = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_bn3")(D)
    D = LeakyReLU(0.2)(D)

    D = Flatten()(D)
    D = Dense(units=128, name=base_name + "_dense1")(D)
    D = LeakyReLU(0.2)(D)
    out = Dense(units=1, activation="sigmoid", name=base_name + "_out")(D)
    network = Network(in_D, out, name=base_name)
    return network


def residual_block(x, base_name, block_num, initializer):
    y = Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv1")(x)
    y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn1")(y)
    y = Activation("relu")(y)
    y = Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False,
               name=base_name + "_resblock" + str(block_num) + "_conv2")(y)
    y = BatchNormalization(momentum=0.9, epsilon=1e-5, name=base_name + "_resblock" + str(block_num) + "_bn2")(y)
    return Add()([x, y])


def save_weights(model, path, counter):
    filename = str(counter) + ".hdf5"
    output_path = os.path.join(path, filename)
    model.save_weights(output_path)
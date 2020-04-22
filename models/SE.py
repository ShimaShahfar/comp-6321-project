import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def Global_Average_Pooling(x):
    # return global_avg_pool(x, name='Global_avg_pooling')
    return tf.reduce_mean(x, axis=[1, 2])


def Fully_connected(x, units, name):
    return layers.Dense(units=units, use_bias=True, name=name)(x)


def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(
            squeeze, units=out_dim / ratio, name=layer_name + "_fully_connected1"
        )
        excitation = layers.ReLU()(excitation)
        excitation = Fully_connected(
            excitation, units=out_dim, name=layer_name + "_fully_connected2"
        )
        excitation = tf.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation

    return scale


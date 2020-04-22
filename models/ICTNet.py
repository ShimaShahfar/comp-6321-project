import os
import numpy as np
from .SE import Squeeze_excitation_layer as se_layer
from tensorflow.keras import layers, models


def preact_conv(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.2, name=""):
    bn = layers.BatchNormalization(fused=True, name=name + "_bn")(inputs)
    preact = layers.ReLU()(bn)

    conv = layers.Conv2D(
        filters=n_filters, kernel_size=kernel_size, padding="same", activation=None, name=name + "_conv"
    )(preact)

    if dropout_p != 0.0:
        conv = layers.Dropout(dropout_p)(conv)
    return conv


def DenseBlock(stack, n_layers, growth_rate, dropout_p, path_type, out_dim, name):
    new_features = []
    for j in range(n_layers):
        # Compute new feature maps
        layer = preact_conv(stack, growth_rate, dropout_p=dropout_p, name=name + f"_preact{j}")
        new_features.append(layer)
        # stack new layer
        stack = layers.concatenate([stack, layer], axis=-1)
    new_features = layers.concatenate(new_features, axis=-1)

    # Special block for SE
    if path_type == "down":
        stack = se_layer(stack, out_dim, 1, (name + path_type))
    else:
        new_features = se_layer(new_features, out_dim, 1, (name + path_type))

    return stack, new_features


def TransitionLayer(inputs, n_filters, dropout_p=0.2, compression=1.0, name=""):
    if compression < 1.0:
        n_filters = int(n_filters * compression)
    l = preact_conv(inputs, n_filters, kernel_size=[1, 1], dropout_p=dropout_p, name=name)
    l = layers.AveragePooling2D([2, 2], strides=[2, 2])(l)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2, name=""):
    l = preact_conv(inputs, n_filters, kernel_size=[1, 1], dropout_p=dropout_p, name=name)
    l = layers.MaxPool2D([2, 2], strides=[2, 2])(l)
    return l


def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, name):
    # Upsample
    l = layers.Conv2DTranspose(
        n_filters_keep, kernel_size=[3, 3], strides=[2, 2], padding="same", name=name+"_deconv"
    )(block_to_upsample)
    # Concatenate with skip connection
    l = layers.concatenate([l, skip_connection], axis=-1)
    return l


def build_fc_densenet(
    inputs,
    preset_model="FC-DenseNet56",
    n_filters_first_conv=48,
    n_pool=5,
    growth_rate=12,
    n_layers_per_block=4,
    dropout_p=0.2,
):
    if preset_model == "FC-DenseNet56":
        n_pool = 5
        growth_rate = 12
        n_layers_per_block = 4
    elif preset_model == "FC-DenseNet67":
        n_pool = 5
        growth_rate = 16
        n_layers_per_block = 5
    elif preset_model == "FC-DenseNet103":
        n_pool = 5
        growth_rate = 16
        n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    if type(n_layers_per_block) == list:
        assert len(n_layers_per_block) == 2 * n_pool + 1
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    # We perform a first convolution.
    stack = layers.Conv2D(
        filters=n_filters_first_conv,
        kernel_size=[3, 3],
        padding="same",
        name="first_conv",
    )(inputs)

    n_filters = n_filters_first_conv

    # Downsampling path
    skip_connection_list = []

    for i in range(n_pool):
        n_filters += growth_rate * n_layers_per_block[i]
        # Dense Block
        stack, _ = DenseBlock(
            stack,
            n_layers_per_block[i],
            growth_rate,
            dropout_p,
            "down",
            n_filters,
            name="down_denseblock%d" % (i + 1),
        )

        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(
            stack, n_filters, dropout_p, name="transitiondown%d" % (i + 1)
        )

    skip_connection_list = skip_connection_list[::-1]

    # Bottleneck Dense Block
    out_dim = n_layers_per_block[n_pool] * growth_rate
    # We will only upsample the new feature maps
    stack, block_to_upsample = DenseBlock(
        stack,
        n_layers_per_block[n_pool],
        growth_rate,
        dropout_p,
        "bottlneck",
        out_dim,
        name="bottle_denseblock%d" % (n_pool + 1),
    )

    # Upsampling path
    for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(
            block_to_upsample,
            skip_connection_list[i],
            n_filters_keep,
            name="transitionup%d" % (n_pool + i + 1),
        )

        # Dense Block
        out_dim = n_layers_per_block[n_pool + i + 1] * growth_rate
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlock(
            stack,
            n_layers_per_block[n_pool + i + 1],
            growth_rate,
            dropout_p,
            "up",
            out_dim,
            name="up_denseblock%d" % (n_pool + i + 2),
        )

    return stack


def ICTNet(input_size=(224, 224, 3), num_classes=1):
    inputs = layers.Input(input_size)
    body = build_fc_densenet(inputs)
    outputs = layers.Conv2D(
        num_classes, [1, 1], activation="softmax", name="probabilities"
    )(body)
    return models.Model(inputs, outputs)

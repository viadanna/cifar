# -*- coding: utf-8 -*-
""" Pretrained inception v3

# Reference

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

"""
from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.backend import tf as ktf
from keras.applications import InceptionV3
from keras import layers


def inception_pretrained():
    """
        Build pretrained inception_v3 model
        Resize images before applying
        Includes trainable layers.
    """
    inputs = layers.Input(shape=(None, None, 3))
    # Resize images
    inputs = layers.Lambda(lambda image: ktf.image.resize_images(image, (139, 139)))(inputs)
    # Build model
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(139, 139, 3),
        input_tensor=inputs)
    for layer in base_model.layers:
        layer.trainable = False
    base_model.trainable = False
    # Trainable layers
    model = layers.Flatten()(base_model.output)
    model = layers.Dense(384, activation='relu')(model)
    model = layers.Dropout(0.5)(model)
    model = layers.Dense(128, activation='relu')(model)
    model = layers.Dropout(0.5)(model)
    model = layers.Dense(10, activation='softmax')(model)
    model = Model(inputs=base_model.input, outputs=model)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

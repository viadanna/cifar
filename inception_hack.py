# -*- coding: utf-8 -*-
"""Pre-trained Inception V3 model for Keras.

Added a slight hack to evaluate only the inception model in CPU.

# Reference

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

"""
from __future__ import print_function
from __future__ import absolute_import

import json
import os
import numpy as np

from keras.layers import Input, Lambda, Flatten, Dense, Dropout
from keras.applications import InceptionV3
from keras.backend import tf as ktf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model

# pylint: disable=missing-docstring,invalid-name,bare-except,W0621
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Stop TensorFlow warnings


def get_inception():
    inputs = Input(shape=(None, None, 3))
    # Resize images
    inputs = Lambda(lambda image: ktf.image.resize_images(image, (139, 139)))(inputs)
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(139, 139, 3),
        input_tensor=inputs)
    base_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    print(base_model.summary())
    return base_model


def evaluate_inception(x, batch_size):
    """ Use Inception to evaluate dataset """
    with ktf.device('/cpu:0'):  # Use CPU as it won't fit my GPU
        model = get_inception()
        return model.predict(
            x=x,
            batch_size=batch_size,
            verbose=1)


def get_evaluated(x, xname, batch_size):
    """ Tries to use saved evaluation """
    try:
        return np.load(xname + '.npy')
    except:
        print('Evaluating {} on inception'.format(xname))
        x = evaluate_inception(x, batch_size)
        with open(xname + '.npy', 'w') as f:
            np.save(f, x)
        return x


if __name__ == '__main__':
    import argparse
    from preprocessing import get_datasets

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--augment', type=int, default=40000)
    args = parser.parse_args()
    if args.augment > 120000:
        raise NotImplementedError('Cannot use generator with this hack.')

    print('Loading datasets')
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_datasets(args.augment)

    BATCH_SIZE = 100
    FILENAME = 'hack_{}'.format(args.augment)

    try:
        # Tries to load the model
        model = load_model('output/{}.h5'.format(FILENAME))
        print('Model loaded')
    except:
        # Build the model to be trained
        X_train = get_evaluated(X_train, 'train_{}'.format(args.augment), BATCH_SIZE)
        X_valid = get_evaluated(X_valid, 'valid', BATCH_SIZE)
        inputs = Input(shape=X_train[0].shape)
        model = Flatten()(inputs)
        model = Dense(384, activation='relu')(model)
        model = Dropout(0.5)(model)
        model = Dense(128, activation='relu')(model)
        model = Dropout(0.5)(model)
        model = Dense(10, activation='softmax')(model)
        model = Model(inputs=inputs, outputs=model)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
        # Training callbacks
        callbacks = [
            EarlyStopping(patience=10),
            TensorBoard(log_dir='output/logs_{}'.format(FILENAME)),
            ModelCheckpoint(filepath='output/{}.h5'.format(FILENAME), save_best_only=True),
        ]
        # Run training
        print('Training model')
        model.fit(
            x=X_train,
            y=y_train,
            batch_size=BATCH_SIZE,
            validation_data=(X_valid, y_valid),
            epochs=1000,
            callbacks=callbacks)
        # Reload best model
        model = load_model('output/{}.h5'.format(FILENAME))
    finally:
        # Evaluate test dataset
        print('Evaluating test dataset')
        X_test = get_evaluated(X_test, 'test', BATCH_SIZE)
        loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        prediction = model.predict(X_test)

        # Save results
        with open('output/{}.json'.format(FILENAME), 'w') as f:
            json.dump({
                'test_accuracy': acc,
                'test_loss': loss,
                'y_true': y_test.tolist(),
                'y_pred': prediction.tolist(),
            }, f)

        print('\nTest Loss: {:.4} | Accuracy: {:.4}'.format(loss, acc))

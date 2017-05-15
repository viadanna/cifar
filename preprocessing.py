"""
    Download and Preprocess cifar10 dataset
"""
from __future__ import print_function
import os
try:
    # python2.7
    import cPickle as pickle
except ImportError:
    # python3.5
    import pickle

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# pylint: disable=invalid-name,line-too-long

if not os.path.exists('input'):
    os.mkdir('input')

if not os.path.exists('output'):
    os.mkdir('output')

# Keras image generator
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True)


def preprocessing(X, y):
    """
        Mean normalization of features
        One-Hot encoding labels
    """
    # Pylint can't find np.float32
    # pylint: disable=no-member
    X = X.astype(np.float32)
    # pylint: enable=no-member
    X -= np.mean(X)
    X /= np.max(X) - np.min(X)
    return X, to_categorical(y)


def get_datasets(train_size=40000):
    """ Returns cifar10 train, validation and test datasets """
    filename = 'input/train_' + str(train_size) + '.p'

    # Check for cached data
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data['X_train'], data['y_train'], data['X_valid'], data['y_valid'], data['X_test'], data['y_test']
    except IOError:
        # Load original dataset using keras
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # Split validation dataset
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              stratify=y_train,
                                                              test_size=0.2,
                                                              random_state=42)

        if train_size > X_train.shape[0]:
            # Augment if required
            print('Augmenting original CIFAR10 dataset')
            X_train, y_train = augment(X_train, y_train, train_size)
        elif train_size < X_train.shape[0]:
            # Reduce if required
            X_train, y_train = X_train[:train_size], y_train[:train_size]

        # Preprocess
        X_train, y_train = preprocessing(X_train, y_train)
        X_valid, y_valid = preprocessing(X_valid, y_valid)
        X_test, y_test = preprocessing(X_test, y_test)

        # Cache preprocessed
        with open(filename, 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


def augment(X, y, size):
    """ Augment given dataset to given size """
    # Keras makes it so easy
    X_aug, y_aug = X, y
    for X_batch, y_batch in tqdm(datagen.flow(X, y, 1000)):
        X_aug = np.vstack((X_aug, X_batch))
        y_aug = np.vstack((y_aug, y_batch))
        if (X_aug.shape[0]) >= size:
            break

    return X_aug, y_aug

# pylint: disable=missing-docstring, invalid-name
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, Input, Reshape, concatenate
from keras.models import Sequential, Model

N_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def simple(n_classes=N_CLASSES, input_shape=INPUT_SHAPE, max_pool=False, keep_prob=1.0):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, input_shape=input_shape, activation='relu'))
    if max_pool:
        model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    if keep_prob < 1.0:
        model.add(Dropout(rate=keep_prob))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def simple_reg(n_classes=N_CLASSES, input_shape=INPUT_SHAPE):
    return simple(n_classes, input_shape, max_pool=True, keep_prob=0.5)


def deep(n_classes=N_CLASSES, input_shape=INPUT_SHAPE):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def lenet(n_classes=N_CLASSES, input_shape=INPUT_SHAPE, keep_prob=1.0):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, input_shape=input_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    if keep_prob < 1.0:
        model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    if keep_prob < 1.0:
        model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def lenet_reg(n_classes=N_CLASSES, input_shape=INPUT_SHAPE):
    return lenet(n_classes, input_shape, keep_prob=0.5)


def deeper(n_classes=N_CLASSES, input_shape=INPUT_SHAPE):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def mininception(n_classes=10, input_shape=(32, 32, 3)):
    # As shown in:
    # https://arxiv.org/pdf/1409.4842.pdf
    def module(input_layer, depth):
        # 1x1
        c1 = Conv2D(depth, 1, activation='relu')(input_layer)

        # 1x1 then 3x3
        c2 = Conv2D(depth, 1, activation='relu')(input_layer)
        c2 = Conv2D(depth, 3, activation='relu', padding='same')(c2)

        # 1x1 then 5x5
        c3 = Conv2D(depth, 1, activation='relu')(input_layer)
        c3 = Conv2D(depth, 5, activation='relu', padding='same')(c3)

        # Max pooling then 1x1
        # m = MaxPool2D((2, 2), padding='same')(x)
        # m = Conv2D(8, 1, activation='relu')(m)

        # Concatenate everything
        output = concatenate([c1, c2, c3], axis=-1)

        return output

    x = Input(input_shape)
    inception_a = module(x, depth=8)
    inception_b = module(inception_a, depth=16)

    pooled_a = MaxPool2D((2, 2))(inception_b)

    inception_c = module(pooled_a, depth=32)
    inception_d = module(inception_c, depth=64)
    pooled_b = MaxPool2D((2, 2))(inception_d)

    inception_e = module(pooled_b, depth=128)
    inception_f = module(inception_e, depth=256)
    pooled_c = MaxPool2D((2, 2))(inception_f)

    output = Reshape((4 * 4 * 256 * 3,))(pooled_c)
    output = Dense(384, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)

    output = Dense(n_classes, activation='softmax')(output)

    model = Model(inputs=x, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

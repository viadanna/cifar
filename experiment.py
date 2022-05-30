# pylint: disable=missing-docstring,invalid-name,line-too-long, bare-except
import argparse
import json

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model

from preprocessing import get_datasets, datagen
from models import simple, simple_reg, deep, lenet, lenet_reg, deeper, mininception
from inception_v3 import inception_pretrained

model_map = {
    'simple': simple,
    'simple_reg': simple_reg,
    'deep': deep,
    'lenet': lenet,
    'lenet_reg': lenet_reg,
    'deeper': deeper,
    'mininception': mininception,
    'inception': inception_pretrained,
}

def train(args):
    # Train given model
    mm = model_map[args.model]()
    # Callbacks for early stopping and tensorboard logging
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1),
        TensorBoard(f'output/{get_model_name(args)}_logs', write_graph=True),
        ModelCheckpoint(filepath=get_model_path(args), save_best_only=True),
    ]

    if args.augment <= 120000:
        # Couldn't save a pickle bigger than that
        X_train, y_train, X_valid, y_valid, _, _ = get_datasets(args.augment)
        mm.fit(
            x=X_train,
            y=y_train,
            batch_size=args.batch,
            epochs=args.epochs,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks)
    else:
        # Generate agumented dataset as the model is trained
        X_train, y_train, X_valid, y_valid, _, _ = get_datasets()
        mm.fit_generator(
            generator=datagen.flow(X_train, y_train, batch_size=args.batch),
            steps_per_epoch=args.augment // args.batch,
            epochs=args.epochs,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks)

    # Reload best model
    return load_model(get_model_path(args))


def evaluate(model, args):
    """ Evaluates the test dataset """
    X_test, y_test = get_datasets(args.augment)[4:]
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    prediction = model.predict(X_test)
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'y_true': y_test.tolist(),
        'y_pred': prediction.tolist(),
    }


def get_model_name(args):
    return f'{args.model}_{args.augment}'


def get_model_path(args):
    return f'output/{get_model_name(args)}.h5'


def get_model(args):
    try:
        return load_model(get_model_path(args))
    except:
        return train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CIFAR-10 experiments')
    parser.add_argument('model', help='Which model(s) to run')
    parser.add_argument('-a', '--augment', type=int, default=40000, help='Training dataset augmentation.')
    parser.add_argument('-b', '--batch', type=int, default=128, help='Batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('-p', '--patience', type=int, default=5, help='Early Stopping patience.')
    arguments = parser.parse_args()

    model = get_model(arguments)
    results = evaluate(model, arguments)
    # Save results
    with open(f'output/{get_model_name(arguments)}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)

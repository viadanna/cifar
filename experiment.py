# pylint: disable=missing-docstring,invalid-name,line-too-long
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CIFAR-10 experiments')
    parser.add_argument('model', help='Which model(s) to run')
    parser.add_argument('-a', '--augment', type=int, default=40000, help='Training dataset augmentation.')
    parser.add_argument('-b', '--batch', type=int, default=128, help='Batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('-p', '--patience', type=int, default=5, help='Early Stopping patience.')
    args = parser.parse_args()

    model_name = '{}_{}'.format(args.model, str(args.augment))
    checkpoint_path = 'output/{}.h5'.format(model_name)
    logs_path = 'output/{}_logs'.format(model_name)

    try:
        # Tries to used saved model
        model = load_model(checkpoint_path)
        _, _, _, _, X_test, y_test = get_datasets(args.augment)
    except:
        # Train given model
        model = model_map[args.model]()
        # Callbacks for early stopping and tensorboard logging
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1),
            TensorBoard(logs_path, write_graph=True),
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True),
        ]

        if args.augment <= 120000:
            # Couldn't save a pickle bigger than that
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_datasets(args.augment)
            history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=args.batch,
                epochs=args.epochs,
                validation_data=(X_valid, y_valid),
                callbacks=callbacks)
        else:
            # Generate agumented dataset as the model is trained
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_datasets()
            history = model.fit_generator(
                generator=datagen.flow(X_train, y_train, batch_size=args.batch),
                steps_per_epoch=args.augment // args.batch,
                epochs=args.epochs,
                validation_data=(X_valid, y_valid),
                callbacks=callbacks)

        # Reload best model
        model = load_model(checkpoint_path)
    finally:
        # Evaluate test dataset
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        prediction = model.predict(X_test)

        # Save results
        with open('output/{}.json'.format(model_name), 'w') as f:
            json.dump({
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'y_true': y_test.tolist(),
                'y_pred': prediction.tolist(),
            }, f)

""" Tests for training and evaluating a Simple model """
from experiment import evaluate, get_model, train


class Args:
    """ Arguments for simple model training """
    model = 'simple'
    patience = 1
    augment = 40000
    batch = 128
    epochs = 1

def test_train_model():
    """ Tests training a simple model """
    assert train(Args)

def test_evaluate_model():
    """ Tests evaluating a model """
    model = get_model(Args)
    assert evaluate(model, Args)

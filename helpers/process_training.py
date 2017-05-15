""" Converts TensorFlow summaries to pandas DataFrame """
# pylint: disable=invalid-name,no-name-in-module

from __future__ import print_function
from glob import glob
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd


def read_summary(path):
    """ Reads a single summary and returns a list """
    acc, loss = [], []
    val_acc, val_loss = [], []
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag == 'loss':
                loss.append(v.simple_value)
            elif v.tag == 'acc':
                acc.append(v.simple_value)
            elif v.tag == 'val_loss':
                val_loss.append(v.simple_value)
            elif v.tag == 'val_acc':
                val_acc.append(v.simple_value)
    epochs = len(acc)
    model = path.split('/')[-2].replace('_logs', '')
    model, size = model.rsplit('_', 1)
    result = [(model, size, epoch, acc[epoch], loss[epoch], val_acc[epoch], val_loss[epoch])
              for epoch in range(epochs)]
    return result


def get_summaries(path='output/*logs'):
    """ Process all summaries in output """
    data = []
    for path in glob(path):
        for fn in glob(path + '/*'):
            data += read_summary(fn)
    data = pd.DataFrame(
        data=data,
        columns=['model', 'size', 'epoch', 'acc', 'loss', 'val_acc', 'val_loss'])
    return data


if __name__ == '__main__':
    print(get_summaries())

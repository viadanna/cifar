""" Converts all JSON output to pandas DataFrame """
# pylint: disable=invalid-name,no-name-in-module

from __future__ import print_function
from glob import glob
import json
import pandas as pd


def read_json(filename):
    """ Read a single JSON adding model name and size """
    with open(filename) as f:
        data = json.load(f)
    model, size = filename.rsplit('_', 1)
    data['model'] = model.split('/')[-1]
    data['size'] = int(size.strip('.json'))
    return data


def get_results(path='output/*json'):
    """ Read all JSON in output/ """
    return pd.DataFrame([read_json(f) for f in glob(path)])


if __name__ == '__main__':
    print(get_results()[['model', 'size', 'test_accuracy']])

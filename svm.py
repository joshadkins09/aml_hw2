#!/usr/bin/env python

import fileinput
import sys
from pprint import pprint

import numpy as np

###############################################################################
# load data
###############################################################################
'''
0:  age: continuous.
2:  fnlwgt: continuous.
4:  education-num: continuous.
10: capital-gain: continuous.
11: capital-loss: continuous.
12: hours-per-week: continuous.
'''
CONTINUOUS_COLUMNS = {0, 2, 4, 10, 11, 12}


def convert_label(value):
    if value == '<=50K':
        return -1
    elif value == '>50K':
        return 1
    else:
        print('bad label "{}"'.format(value))
        sys.exit(2)


def get_data_from_line(split_line, include_labels):
    data = [
        float(field) for cnt, field in enumerate(split_line)
        if cnt in CONTINUOUS_COLUMNS
    ]
    if include_labels:
        return convert_label(split_line[-1]), data
    else:
        return data


def load_data_from_file(filename, include_labels):
    return [
        get_data_from_line(line.rstrip('\n').split(', '), include_labels)
        for line in fileinput.input(filename)
    ]


def load_data():
    train_labels, train_data = zip(*load_data_from_file('train.txt', True))
    test_data = load_data_from_file('test.txt', False)
    return (np.array(train_labels), np.array(train_data), np.array(test_data))


###############################################################################


def f(x, a=0, b=0):
    return np.multiply(a, x) + b


def cost_funtion(x, y, a, b):
    foo = 1 - np.multiply(y, f(x, a, b))
    # r1 = np.maximum()
    print(foo)


def main():
    print('running')
    train_labels, train_data, test_data = load_data()
    print(len(train_labels), len(train_data), len(test_data))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

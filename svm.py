#!/usr/bin/env python

import fileinput
import sys

from pprint import pprint as print

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


def get_fields_from_line(line):
    return [float(field) for count, field in enumerate(line.split(', '))
            if count in CONTINUOUS_COLUMNS]


def load_data_from_file(filename):
    return [get_fields_from_line(line) for line in fileinput.input(filename)]


def load_data():
    train = load_data_from_file('train.txt')
    test = load_data_from_file('test.txt')
    return train, test


def main():
    print('running')
    train, test = load_data()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

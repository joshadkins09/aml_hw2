#!/usr/bin/env python

import fileinput
import random
import sys

import numpy as np

###############################################################################
# utils for splitting testing from training data from single dataset
###############################################################################


def calculate_pivot(data_size, percent_to_use_for_test):
    return data_size * (100 - percent_to_use_for_test) // 100


def split_data(data, pivot, shuffle=False):
    if shuffle:
        np.random.shuffle(data)
    training, test = data[:pivot], data[pivot:]
    return training, test


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


def load_train_data(filename, include_labels=True):
    labels_and_data = load_data_from_file(filename, include_labels)
    pivot = calculate_pivot(len(labels_and_data), 10)
    train, test = split_data(labels_and_data, pivot, shuffle=False)
    train_labels, train_data = zip(*train)
    test_labels, test_data = zip(*test)

    # center for zero mean and scale for unit variance
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    return (np.array(train_labels), np.array(train_data),
            np.array(test_labels), np.array(test_data))


def load_test_data(filename, include_labels=True):
    data = load_data_from_file(filename, include_labels)
    # should we preprocess the test data? bad data if we don't -jadkins
    data = preprocess(data)
    return np.array(data)


###############################################################################


def preprocess(data):
    mean = np.mean(data, axis=0)
    centered = data - mean
    std = np.std(centered, axis=0)
    scaled = centered / std
    return scaled


###############################################################################
# misc utils
###############################################################################


def f(x, a, b):
    return np.multiply(a, x) + b


def signof(val):
    if val < 0:
        return -1
    return 1


###############################################################################
# svm classifier
###############################################################################


class SVM:
    def __init__(self, lam, m=1, n=2):
        self.lam = lam
        self.m = m
        self.n = n
        self.a_initial = np.random.rand(6)
        self.b_initial = np.random.rand(1)[0]
        self.a = self.a_initial
        self.b = self.b_initial
        self.accuracy = list()

    def reset(self):
        self.a = 0
        self.b = 0
        self.accuracy = list()

    def f(self, x):
        return np.dot(self.a, x) + self.b

    def eta(self, current_step):
        return (self.m / (current_step + self.n))

    def gradient_of_cost(self, x, y, a, b):
        v = y * self.f(x)
        if v >= 1:
            grad_a = self.lam * a
            grad_b = 0
        else:
            grad_a = self.lam * a - (y * x)
            grad_b = -y
        return grad_a, grad_b

    def step(self, x_batch, y_batch, a, b, step_number):
        grad_a, grad_b = self.gradient_of_cost(x_batch, y_batch, a, b)
        a_new = a - self.eta(step_number) * grad_a
        b_new = b - self.eta(step_number) * grad_b
        return (a_new, b_new)

    def search_for_min_cost_params(self,
                                   data,
                                   labels,
                                   step,
                                   holdout_data=None,
                                   holdout_labels=None,
                                   sac=None):
        n = random.randint(0, len(data) - 1)
        self.a, self.b = self.step(data[n], labels[n], self.a, self.b, step)

        if sac is not None:
            if step % 30 == 0:
                acc = self.measure_accuracy(self.a, self.b, holdout_data,
                                            holdout_labels)
                sac.append(acc)
            self.accuracy.append(sac)

    def measure_accuracy(self, a, b, holdout_data, holdout_labels):
        res = [signof(self.f(d)) for c, d in enumerate(holdout_data)]
        j = [(i, j) for i, j in zip(res, holdout_labels) if i == j]
        return (len(j) / len(holdout_labels))

    def fit_with_measurements(self,
                              train_data,
                              train_labels,
                              num_seasons=50,
                              num_steps=300):
        self.reset()
        self.a = self.a * np.zeros([len(train_data[0])])

        zipped = list(zip(train_data, train_labels))
        for season in range(num_seasons):
            sac = list()
            np.random.shuffle(zipped)
            holdout_data, holdout_labels = zip(*zipped[:50])
            data, labels = zip(*zipped[50:])

            for step in range(num_steps):
                self.search_for_min_cost_params(
                    data, labels, step, holdout_data, holdout_labels, sac)

    def fit(self, train_data, train_labels, num_seasons=50, num_steps=300):
        self.a = self.a * np.zeros([len(train_data[0])])
        for season in range(num_seasons):
            data, labels = train_data, train_labels
            for step in range(num_steps):
                self.search_for_min_cost_params(data, labels, step)

    def predict(self, test_data):
        return signof(self.f(test_data))


def fmt_for_sub(val):
    if val == 1:
        return '>50K'
    elif val == -1:
        return '<=50K'
    else:
        print('invlaid value')
        sys.exit(3)


def generate_submission(clf, data):
    res = [fmt_for_sub(clf.predict(x)) for x in data]
    with open('submission.txt', 'w') as sub:
        for r in res:
            sub.write('{}\n'.format(r))


def fit_and_predict_real_data(train_data, train_labels, v_data, v_labels):
    svm = SVM(.001, 1, 2)
    svm.a = np.array([
        0.77939948, 0.85269224, 0.21391138, 0.88460607, 0.47162305, 0.54078254
    ])
    svm.b = 0.03354322864033388
    data = np.concatenate((train_data, v_data))
    labels = np.concatenate((train_labels, v_labels))

    svm.fit(data, labels, num_seasons=50)
    real_test_data = load_test_data('test.txt', False)
    generate_submission(svm, real_test_data)


def iter_lambda(train_data, train_labels, v_data, v_labels):
    svm = SVM(1, 1, 2)
    print(svm.a_initial, svm.b_initial)
    for lam in [.0001, .001, .01, .1, 1]:
        svm.lam = lam
        svm.fit_with_measurements(train_data, train_labels)
        predictions = [svm.predict(x) for x in v_data]
        v = [(a, b) for a, b in zip(predictions, v_labels) if a == b]
        print(len(v) / len(v_labels))


def main():
    train_labels, train_data, v_labels, v_data = load_train_data(
        'train.txt', True)

    # iter_lambda(train_data, train_labels, v_data, v_labels)
    fit_and_predict_real_data(train_data, train_labels, v_data, v_labels)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

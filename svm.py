#!/usr/bin/env python
'''
u, c = np.unique(train_labels, return_counts=True)
acc = dict(zip(u, c))# / len(test_data)
print(acc)

Procedure: 4.1 Training an SVM: Overall

Start with a dataset containing N pairs (xi, yi). Each xi is a d- dimensional
feature vector, and each yi is a label, either 1 or −1. Optionally, rescale the
xi so that each component has unit variance. Choose a set of possible values of
the regularization weight λ. Separate the dataset into two sets: test and
training. Reserve the test set. For each value of the regularization weight,
use the training set to estimate the accuracy of an SVM with that λ value,
using cross-validation as in procedure 4.2 and stochastic gradient descent. Use
this information to choose λ0, the best value of λ (usually, the one that
yields the highest accuracy). Now use the training set to fit the best SVM
using λ0 as the regularization constant. Finally, use the test set to compute
the accuracy or error rate of that SVM, and report that

Procedure: 4.2 Training an SVM: estimating the accuracy

Repeatedly: split the training dataset into two components (training and
validation), at random; use the training component to train an SVM; and compute
the accuracy on the validation component. Now average the resulting accuracy
values.

'''

import fileinput
import math
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


def load_data():
    labels_and_data = load_data_from_file('train.txt', True)
    pivot = calculate_pivot(len(labels_and_data), 10)
    train, test = split_data(labels_and_data, pivot, shuffle=False)
    train_labels, train_data = zip(*train)
    test_labels, test_data = zip(*test)

    # center for zero mean and scale for unit variance
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    return (np.array(train_labels), np.array(train_data),
            np.array(test_labels), np.array(test_data))


###############################################################################


def preprocess(data):
    mean = np.mean(data, axis=0)
    centered = data - mean
    std = np.std(centered, axis=0)
    scaled = centered / std
    return scaled


###############################################################################

# def f(x, a=0, b=0):
#     return np.multiply(a, x) + b

# not sure if we ever need the cost function -jadkins
# def cost_function(x, y, a, b):
#     foo = 1 - np.dot(y, f(x, a, b))
#     r1 = np.maximum(0, foo)
#     return r1

def signof(val):
    if val < 0:
        return -1
    return 1


class SVM:
    num_seasons = 50
    m = 1
    n = 2

    def __init__(self, lam):
        self.lam = lam

    def f(self, x, a, b):
        return np.dot(a, x) + b

    def eta(self, current_step):
        return (self.m / (current_step + self.n))

    # this may be private
    def gradient_of_cost(self, x, y, a, b):
        # not sure if this is the correct type of multiplcation -jadkins
        v = y * self.f(x, a, b)
        if v >= 1:
            grad_a = self.lam * a
            grad_b = 0
        else:
            grad_a = self.lam * a - (y * x)
            grad_b = -y
        return grad_a, grad_b

    def step(self, x_batch, y_batch, a_est, b_est, step_number):
        # c = cost_function(x_batch, y_batch, a=a_est, b=b_est)
        grad_a, grad_b = self.gradient_of_cost(x_batch, y_batch, a_est, b_est)
        a_new = a_est - self.eta(step_number) * grad_a
        b_new = b_est - self.eta(step_number) * grad_b
        return (a_new, b_new)

    # def at_each_step(self):
    #     n = random.randint(0, len(x) - 1)  # Batch size of 1.
    #     a_est, b_est = step(x[n], y[n], a_est, b_est, lam, eta)
    #     # Normally we wouldn't collect cost and parameters every iteration,
    #     # but this is a very simple function to learn.
    #     all_a_est.append(a_est)
    #     all_b_est.append(b_est)
    #     # This is the batch training cost...this should be fixed to be
    #     # validation set.
    #     all_costs.append(math.log(cost_function(x, y, a=a_est, b=b_est)))

    def measure_accuracy(self, a, b, holdout_data, holdout_labels):
        # count = 0
        res = [signof(self.f(d, a, b)) for c, d in enumerate(holdout_data)]
        # for predicted, actual in zip(res, holdout_labels):
        #     print(predicted, actual)

        j = [(i, j) for i, j in zip(res, holdout_labels) if i == j]
        print(len(j)/50)
        # for predicted, actual in zip(res, holdout_labels):
        #     print(predicted, actual)
        # for c, d in enumerate(holdout_data):
        #     res = self.f(d, a, b)
        #     print(res, holdout_labels[c])

    def fit(self, train_data, train_labels, num_seasons=50, num_steps=300):
        # tmp
        all_a_est, all_b_est = list(), list()
        a_est = np.zeros([len(train_data[0])])
        b_est = 0

        zipped = list(zip(train_data, train_labels))
        for season in range(num_seasons):
            # definitely should make sure that all this wizardry works -jadkins
            np.random.shuffle(zipped)
            holdout_data, holdout_labels = zip(*zipped[:50])
            data, labels = zip(*zipped[50:])
            for step in range(num_steps):
                # could compute len(data) outside -jadkins
                n = random.randint(0, len(data) - 1)  # Batch size of 1.
                a_est, b_est = self.step(data[n], labels[n], a_est, b_est,
                                         step)
                all_a_est.append(a_est)
                all_b_est.append(b_est)

                if step % 30 == 0:
                    self.measure_accuracy(a_est, b_est, holdout_data,
                                          holdout_labels)


def main():
    print('running')
    train_labels, train_data, test_labels, test_data = load_data()
    svm = SVM(1)
    print('before')
    svm.fit(train_data, train_labels)
    print('after')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

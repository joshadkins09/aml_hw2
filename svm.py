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


# could use sklearn.preprocessing.scale for this -jadkins
def preprocess(data):
    mean = np.mean(data, axis=0)
    centered = data - mean
    std = np.std(centered, axis=0)
    scaled = centered / std
    return scaled


###############################################################################


def f(x, a, b):
    return np.multiply(a, x) + b


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

    def __init__(self, lam, m=6, n=10):
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
        # return .1
        return (self.m / (current_step + self.n))

    # this may be private
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
        # c = cost_function(x_batch, y_batch, a=a, b=b)
        grad_a, grad_b = self.gradient_of_cost(x_batch, y_batch, a, b)
        a_new = a - self.eta(step_number) * grad_a
        b_new = b - self.eta(step_number) * grad_b
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
        res = [signof(self.f(d)) for c, d in enumerate(holdout_data)]
        j = [(i, j) for i, j in zip(res, holdout_labels) if i == j]
        return (len(j) / len(holdout_labels))

    # NOTE: before turning in, need to separate out sgd from fit -jadkins
    def fit(self, train_data, train_labels, num_seasons=50, num_steps=300):
        self.reset()
        self.a = self.a * np.zeros([len(train_data[0])])

        zipped = list(zip(train_data, train_labels))
        for season in range(num_seasons):
            sac = list()
            # definitely should make sure that all this wizardry works -jadkins
            np.random.shuffle(zipped)
            holdout_data, holdout_labels = zip(*zipped[:50])
            data, labels = zip(*zipped[50:])
            for step in range(num_steps):
                # could compute len(data) outside -jadkins
                n = random.randint(0, len(data) - 1)  # Batch size of 1.
                self.a, self.b = self.step(data[n], labels[n], self.a, self.b,
                                           step)

                if step % 30 == 0:
                    acc = self.measure_accuracy(self.a, self.b, holdout_data,
                                                holdout_labels)
                    sac.append(acc)
            self.accuracy.append(sac)

    def fancy_fit(self,
                  train_data,
                  train_labels,
                  num_seasons=50,
                  num_steps=300):
        self.a = self.a * np.zeros([len(train_data[0])])
        for season in range(num_seasons):
            data, labels = train_data, train_labels
            for step in range(num_steps):
                # could compute len(data) outside -jadkins
                n = random.randint(0, len(data) - 1)  # Batch size of 1.
                self.a, self.b = self.step(data[n], labels[n], self.a, self.b,
                                           step)

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


def main():
    # print('running')
    train_labels, train_data, validation_labels, validation_data = load_train_data(
        'train.txt', True)

    # this is for seeing accuracy with varied lambda
    # svm = SVM(1, 1, 2)
    # print(svm.a_initial, svm.b_initial)
    # # print results of training on validation set
    # for lam in [.0001, .001, .01, .1, 1]:
    #     svm.lam = lam
    #     svm.fit(train_data, train_labels)
    #     predictions = [svm.predict(x) for x in validation_data]
    #     v = [(a, b) for a, b in zip(predictions, validation_labels) if a == b]
    #     print(len(v) / len(validation_labels))
    #     # pprint(svm.accuracy)

    # this is for the submission.txt
    # takes lambda, m, n for calculating eta (learning rate)
    svm = SVM(.001, 1, 2)
    svm.a = np.array([0.77939948, 0.85269224, 0.21391138, 0.88460607, 0.47162305, 0.54078254])
    svm.b = 0.03354322864033388
    data = np.concatenate((train_data, validation_data))
    labels = np.concatenate((train_labels, validation_labels))

    svm.fancy_fit(data, labels, num_seasons=50)
    real_test_data = load_test_data('test.txt', False)
    generate_submission(svm, real_test_data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

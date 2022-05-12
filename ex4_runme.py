"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

"""
import numpy as np
from ex4_tools import DecisionStump, decision_boundaries, generate_data
import matplotlib.pyplot as plt
from adaboost import AdaBoost



def Q4_b():
    training_error = []
    test_error = []
    adaBoost = AdaBoost(DecisionStump, num_classifiers)
    adaBoost.train(train_X, train_y)
    num_classifiers_list = np.arange(1, num_classifiers)
    for T in num_classifiers_list:
        training_error.append(adaBoost.error(train_X, train_y, T))
        test_error.append(adaBoost.error(test_X, test_y, T))
    plt.plot(num_classifiers_list, training_error, label='train error')
    plt.plot(num_classifiers_list, test_error, label='test error')
    plt.title('Adaboost error')
    plt.legend()
    plt.show()


def Q4_c():
    adaBoost = AdaBoost(DecisionStump, num_classifiers)
    adaBoost.train(train_X, train_y)
    for i, T in enumerate([5, 10, 50, 100, 200, 500]):
        plt.subplot(2, 3, i + 1)
        decision_boundaries(adaBoost, test_X, test_y, T)
    plt.show()


def Q4_d():
    test_error = []
    adaBoost = AdaBoost(DecisionStump, num_classifiers)
    adaBoost.train(train_X, train_y)
    num_classifiers_list = np.arange(1, num_classifiers)
    for T in num_classifiers_list:
        test_error.append(adaBoost.error(test_X, test_y, T))
        print(T)
    T_hat = np.argmin(test_error) + 1
    decision_boundaries(adaBoost, test_X, test_y, T_hat)
    plt.show()


def Q4_e():
    num_classifiers = 500
    adaBoost = AdaBoost(DecisionStump, num_classifiers)
    D = adaBoost.train(train_X, train_y)
    D = D / np.max(D) * 10
    decision_boundaries(adaBoost, train_X, train_y, num_classifiers, weights=D)
    plt.show()


if __name__ == '__main__':
    num_classifiers = 500
    train_X, train_y = generate_data(5000, 0.)
    test_X, test_y = generate_data(500, 0.)
    Q4_b()
    Q4_c()
    Q4_d()
    Q4_e()


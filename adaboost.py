"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        n = y.shape[0]
        D = np.ones(n) / n
        for i in range(0, self.T):
            print(i)
            self.h[i] = self.WL(D, X, y)
            y_hat = self.h[i].predict(X)
            epsilon = np.sum((np.abs(y_hat - y) / 2) * D)
            self.w[i] = 0.5 * np.log(1.0 / epsilon - 1)
            D *= np.exp((-1) * self.w[i] * y * y_hat)
            D /= np.sum(D)
        return D

    def predict(self, X, max_t):
        """
        :param X: the samples we want to classify.
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X
        """
        y_hat = np.zeros(X.shape[0])
        for t in range(0, max_t):
            y_hat += (self.h[t].predict(X) * self.w[t])

        return np.sign(y_hat)

    def error(self, X, y, max_t):
        """
        :param X: the samples we want to classify.
        :param y: the labels of the samples
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions
        """
        y_hat = self.predict(X, max_t)
        return sum(abs(y - y_hat)/2) / y.shape[0]


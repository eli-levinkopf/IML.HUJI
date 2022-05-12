from __future__ import annotations
from traceback import print_tb
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ , self.error_ = None, None, 1, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.j_, self.error_, self.threshold_ = self.__fit(X, y)
        negative_j, negative_error, negative_threshold = self.__fit(X, y, -1)
        if negative_error < self.error_:
            self.threshold_, self.j_, self.sign_ = negative_threshold, negative_j, 1


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.sign_ * ((X[:, self.j_] >= self.threshold_) * 2 - 1)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        D = np.abs(labels)
        y = np.sign(labels)
        idx = np.argsort(values)
        X, y, D = values[idx], y[idx], D[idx]
        minimal_theta_loss = np.sum(D[y == sign])
        losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(D * (y * sign)))
        min_loss_idx = np.argmin(losses)
        theta_arr = np.concatenate([[-np.inf], (X[1:] + X[:-1]) / 2, [np.inf]])
        return theta_arr[min_loss_idx], losses[min_loss_idx]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return np.sum(np.abs(y[np.sign(y) != np.sign(y_pred)]))

    def __fit(self, X: np.ndarray, y: np.ndarray, sign: int=1):
        errors_arr, thresholds_arr = np.array([]), np.array([])
        for i in range(X.shape[1]):
            threshold, error = self._find_threshold(X[:, i], y, sign)
            thresholds_arr = np.append(thresholds_arr, threshold)
            errors_arr = np.append(errors_arr, error)
        i = np.argmin(errors_arr)
        return i,  errors_arr[i], thresholds_arr[i]
        
from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    train_error, validation_error = .0, .0
    remainder_ = np.remainder(np.arange(y.size), cv)
    for i in range(cv):
        train_set_x, train_set_y = X[remainder_ != i], y[remainder_ != i]
        validation_set_x, validation_set_y = X[remainder_ == i], y[remainder_ == i]
        estimator.fit(train_set_x, train_set_y)
        loss_train = scoring(estimator.predict(train_set_x), train_set_y)
        loss_validation = scoring(estimator.predict(validation_set_x), validation_set_y)
        train_error += loss_train
        train_error /= cv
        validation_error += loss_validation
        validation_error /= cv
    return (train_error, validation_error)
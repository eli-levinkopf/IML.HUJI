import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import plotly
import sklearn
from sklearn.metrics import auc
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_list = []

    def helper(solver: GradientDescent, weights: np.ndarray, val: np.ndarray, grad: np.ndarray, t: int, eta: float,
               delta: float):
        values.append(val)
        weights_list.append(weights)

    return helper, values, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        fixed_lr = FixedLR(eta)
        callback1, values1, weights1 = get_gd_state_recorder_callback()
        solver1 = GradientDescent(fixed_lr, callback=callback1)
        solver1.fit(L1(init), X=None, y=None)
        plot_descent_path(L1, np.concatenate(weights1, axis=0).reshape(len(weights1), len(init)),
                          f"Descent trajectory, L1, eta = {eta}").show()

        callback2, values2, weights2 = get_gd_state_recorder_callback()
        solver2 = GradientDescent(fixed_lr, callback=callback2)
        solver2.fit(L2(init), X=None, y=None)
        plot_descent_path(L2, np.concatenate(weights2, axis=0).reshape(len(weights2), len(init)),
                          f"Descent trajectory, L2, eta = {eta}").show()

        axis_x = list(range(len(weights1)))

        go.Figure(go.Scatter(x=axis_x, y=values1, mode='markers', name=f'Convergence Rate L1. eta = {eta}')) \
            .update_layout(title=f'Convergence Rate L1. eta = {eta}').show()

        axis_x = list(range(len(weights2)))

        go.Figure(go.Scatter(x=axis_x, y=values2, mode='markers', name=f'Convergence Rate L2. eta = {eta}')) \
            .update_layout(title=f'Convergence Rate L2. eta = {eta}').show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = plotly.subplots.make_subplots(rows=2, cols=2)
    l1 = L1(init)
    for i, gamma in enumerate(gammas):
        exp_lr = ExponentialLR(base_lr=eta, decay_rate=gamma)
        callback, values, weights = get_gd_state_recorder_callback()
        solver = GradientDescent(learning_rate=exp_lr, callback=callback)
        solver.fit(f=l1, X=None, y=None)

        fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode='markers', name=f'decay rate: {gamma}'),
                      row=i % 2 + 1, col=i // 2 + 1)

    fig.update_layout(title=f'Convergence Rate for L1, with different exponentially decaying learning rates')

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    exp_lr = ExponentialLR(base_lr=eta, decay_rate=0.95)
    callback1, values1, weights1 = get_gd_state_recorder_callback()
    solver1 = GradientDescent(exp_lr, callback=callback1)
    solver1.fit(L1(init), X=None, y=None)
    plot_descent_path(L1, np.concatenate(weights1, axis=0).reshape(len(weights1), len(init)),
                      f"Descent trajectory for L1, with eta = {eta}, decay rate = 0.95").show()

    exp_lr = ExponentialLR(base_lr=eta, decay_rate=0.95)
    callback2, values2, weights2 = get_gd_state_recorder_callback()
    solver2 = GradientDescent(exp_lr, callback=callback2)
    solver2.fit(L2(init), X=None, y=None)
    plot_descent_path(L2, np.concatenate(weights2, axis=0).reshape(len(weights2), len(init)),
                      f"Descent trajectory for L2, with eta = {eta}, decay rate = 0.95").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression()
    model.fit(X=X_train, y=y_train)
    y_prob = model.predict_proba(X_test)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_prob)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    best_alpha = thresholds[np.argmax(tpr - fpr)]

    best_model = LogisticRegression(alpha=best_alpha)
    best_model.fit(X_train, y_train)
    test_err = best_model.loss(X_test, y_test)
    print(f'Test error of model with threshold = {best_alpha} is {test_err}')

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    for norm in ['l1', 'l2']:
        min_validate_err = 1
        best_lam = 0
        for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:

            train_score, validation_score = \
                cross_validate(LogisticRegression(penalty=norm, lam=lam), X_train, y_train, misclassification_error)
            if validation_score < min_validate_err:
                min_validate_err = validation_score
                best_lam = lam

        model_test_err = LogisticRegression(penalty=norm, lam=best_lam).fit(X_train, y_train).loss(X_test, y_test)

        print(f'for {norm} model - value of selected lambda is: {best_lam}, and model test error is {model_test_err}')


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()

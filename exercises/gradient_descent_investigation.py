import sys
sys.path.append('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI')
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
import plotly
import plotly.graph_objects as go
import sklearn
from sklearn.metrics import auc

fig_idx = 1


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
    weights_array, values_array = [], []
    def inner(solver, weight, value, grad, t, eta, delta):
        weights_array.append(weight)
        values_array.append(value)
    return inner, values_array, weights_array




def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    global fig_idx
    for eta in etas:
        callback1, values_array1, weights_array1 = get_gd_state_recorder_callback()
        fixedlr = FixedLR(eta)
        solver1 = GradientDescent(fixedlr,  callback=callback1)
        solver1.fit(L1(init), X=None, y=None)
        descent_path = np.concatenate(weights_array1, axis=0).reshape(len(weights_array1), len(init))
        fig = plot_descent_path(L1, descent_path, f'| L1 | eta: {eta}')
        fig.write_image(f'/Users/elilevinkopf/Documents/Ex22B/IML/ex6/photos/fig{fig_idx}.png')
        fig_idx+=1

        fig = go.Figure(go.Scatter(x=list(range(len(weights_array1))), y=values_array1, mode='markers', name=f'Convergence Rate L1 | eta: {eta}'))
        fig.update_layout(title=f'Convergence Rate L1. eta: {eta}')
        fig.write_image(f'/Users/elilevinkopf/Documents/Ex22B/IML/ex6/photos/fig{fig_idx}.png')
        fig_idx+=1

        callback2, values_array2, weights_array2 = get_gd_state_recorder_callback()
        solver2 = GradientDescent(fixedlr, callback=callback2)
        solver2.fit(L2(init), X=None, y=None)
        descent_path = np.concatenate(weights_array2, axis=0).reshape(len(weights_array2), len(init))
        fig = plot_descent_path(L1, descent_path, f'| L2 | eta: {eta}')
        fig.write_image(f'/Users/elilevinkopf/Documents/Ex22B/IML/ex6/photos/fig{fig_idx}.png')
        fig_idx+=1

        fig = go.Figure(go.Scatter(x=list(range(len(weights_array2))), y=values_array2, mode='markers', name=f'Convergence Rate L2 | eta: {eta}'))
        fig.update_layout(title=f'Convergence Rate L2. eta: {eta}')
        fig.write_image(f'/Users/elilevinkopf/Documents/Ex22B/IML/ex6/photos/fig{fig_idx}.png')
        fig_idx+=1

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    global fig_idx
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = plotly.subplots.make_subplots(rows=2, cols=2)
    l1 = L1(init)
    for i, gamma in enumerate(gammas):
        callback1, values_array1, weights_array1 = get_gd_state_recorder_callback()
        explr = ExponentialLR(base_lr=eta, decay_rate=gamma)
        solver = GradientDescent(learning_rate=explr, callback=callback1)
        solver.fit(f=l1, X=None, y=None)
        fig.add_trace(go.Scatter(x=list(range(len(values_array1))), y=values_array1, mode='markers', name=f'DR: {gamma}'),row=i % 2 + 1, col=i // 2 + 1)
    fig.update_layout(title=f'Convergence Rate for L1')    
    
    # Plot algorithm's convergence for the different values of gamma
    fig.write_image(f'/Users/elilevinkopf/Documents/Ex22B/IML/ex6/photos/fig{fig_idx}.png')
    fig_idx+=1

    # Plot descent path for gamma=0.95
    callback2, values_array2, weights_array2 = get_gd_state_recorder_callback()
    explr = ExponentialLR(base_lr=eta, decay_rate=0.95)
    solver1 = GradientDescent(explr, callback=callback2)
    solver1.fit(L1(init), X=None, y=None)
    fig = plot_descent_path(L1, np.concatenate(weights_array2, axis=0).reshape(len(weights_array2), len(init)), f"L1 | with eta: {eta}, decay rate = 0.95")
    fig.write_image(f'/Users/elilevinkopf/Documents/Ex22B/IML/ex6/photos/fig{fig_idx}.png')
    fig_idx+=1

    callback3, values_array3, weights_array3 = get_gd_state_recorder_callback()
    explr = ExponentialLR(base_lr=eta, decay_rate=0.95)
    solver2 = GradientDescent(explr, callback=callback3)
    solver2.fit(L2(init), X=None, y=None)
    fig = plot_descent_path(L2, np.concatenate(weights_array3, axis=0).reshape(len(weights_array3), len(init)), f"L2 | with eta: {eta}, decay rate = 0.95")
    fig.write_image(f'/Users/elilevinkopf/Documents/Ex22B/IML/ex6/photos/fig{fig_idx}.png')
    fig_idx+=1


def load_data(path: str = "/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI/datasets/SAheart.data", train_portion: float = .8) -> \
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



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
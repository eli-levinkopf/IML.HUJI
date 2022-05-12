# TODO: Remove before submission
import sys
sys.path.append('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI')
import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    test_error, train_error = np.ndarray(n_learners), np.ndarray(n_learners)
    for i in range(n_learners):
        test_error[i] = adaboost.partial_loss(test_X, test_y, i + 1)
        train_error[i] = adaboost.partial_loss(train_X, train_y, i + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, n_learners)), y=test_error, mode='lines', name=r'$\textbf{testError}$'))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners)), y=train_error, mode='lines', name=r'$\textbf{trainError}$'))

    if not noise:
        fig.update_layout(title=r'$\textbf{Train and test error of AdaBoost in noiseless case}$',
                        xaxis=dict(title='numberOfLearners'), yaxis=dict(title='error value'))
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/AdaBoostError.png')
    else: 
        fig.update_layout(title=r'$\textbf{Train and test error of AdaBoost in noise case}$',
                        xaxis=dict(title='numberOfLearners'), yaxis=dict(title='error value'))
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/AdaBoostErrorNoise.png')


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf'$\textbf{{Decision boundaries for {t} learners}}$' for t in T],
                        horizontal_spacing=.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X :adaboost.partial_predict(X,t), lims[0], lims[1], density=t, showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False, marker=dict(color=train_y,
                         colorscale=custom,line=dict(color="black", width=0.1))),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False, marker=dict(color=test_y,
                         colorscale=custom,line=dict(color="black", width=0.1)))],rows=(i // 2) + 1, cols=(i % 2) + 1)

    if not noise:    
        fig.update_xaxes(visible=False).update_yaxes(visible=False)
        fig.update_layout(title=r"$\textbf{Decision boundaries as a function of number of learners in noiseless case}$", margin=dict(t=100))
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/DecisionBoundaries.png')
    else:
        fig.update_xaxes(visible=False).update_yaxes(visible=False)
        fig.update_layout(title=r"$\textbf{Decision boundaries as a function of number of learners in noise case}$", margin=dict(t=100))
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/DecisionBoundariesNoise.png')


    # Question 3: Decision surface of best performing ensemble
    best_size = np.argmin(test_error)
    accuracy = 1 - test_error[best_size]
    best_size += 1
    fig = go.Figure([decision_surface(lambda X :adaboost.partial_predict(X,t), lims[0], lims[1], density=best_size, showscale=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=test_y, colorscale=custom, line=dict(color="black", width=0.1)))])

    if not noise:  
        fig.update_layout(title=rf'$\textbf{{Decision surface for best performing ensemble in noiseless case}}\\\
                                        \text{{best_size}} = {best_size}'rf'\text{{, accuracy =}}{accuracy}$')
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/BestEnsemble.png')
    else: 
        fig.update_layout(title=rf'$\textbf{{Decision surface for best performing ensemble in noise case}}\\\
                                            \text{{best_size}} = {best_size}'rf'\text{{, accuracy =}}{accuracy}$')
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/BestEnsembleNoise.png')


    # Question 4: Decision surface with weighted samples
    D = adaboost.D_
    D = D / np.max(D) * 5
    fig = go.Figure([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y, colorscale=custom, line=dict(color="black", width=1), size=D))])
    if not noise:
        fig.update_layout(title=rf'$\textbf{{Decision surface with weighted samples in noiseless case}}$')
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/decisionWithWeighted.png')
    else:
        fig.update_layout(title=rf'$\textbf{{Decision surface with weighted samples in noise case}}$')
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex4/plots/decisionWithWeightedNoise.png')



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
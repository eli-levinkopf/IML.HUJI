from __future__ import annotations
import sys
sys.path.append('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI')

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x -2)
    X = np.linspace(-1.2, 2, n_samples)
    y = f(X) + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y  = split_train_test(pd.DataFrame(X), pd.Series(y), 2/3)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=X, y=f(X), mode='markers', marker=dict(color='black'), showlegend=False))
    fig1.add_trace(go.Scatter(x=train_X[0], y=train_y, mode='markers', marker=dict(color='blue'),name="Train"))
    fig1.add_trace(go.Scatter(x=test_X[0], y=test_y, mode='markers', marker=dict(color='green'), name='Test'))
    fig1.update_layout(title_text='Train and Test Samples', height=400, width=600)
    if noise == 5:
        fig1.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q1.png')
    elif noise == 0:
        fig1.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q4.1.png')
    else:
        fig1.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q5.1.png')

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    validation_array, training_array = [], []
    for i in range(11):
        p = PolynomialFitting(i)
        tmp_validation, tmp_training = cross_validate(p, train_X[0].to_numpy(), train_y.to_numpy(), mean_square_error)
        training_array.append(tmp_validation)
        validation_array.append(tmp_training)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(name='trainError', x=list(range(11)), y=training_array, mode='markers+lines',  marker_color='rgb(152,171,150)'))
    fig2.add_trace(go.Scatter(name='validationError', x=list(range(11)), y=validation_array, mode='markers+lines', marker_color='rgb(25,115,132)'))
    fig2.update_layout(title='average training and validation errors of poly of degree k', xaxis_title='Degree k', yaxis_title='MSE', height=300, width=600)
    if noise == 5:
        fig2.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q2.png')
    elif noise == 0:
        fig2.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q4.2.png')
    else:
        fig2.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q5.2.png')


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    idx = np.argmin(np.array(validation_array))
    k = np.array(list(range(11)))[idx]
    p = PolynomialFitting(k)
    p.fit(train_X[0].to_numpy(), train_y.to_numpy())
    if noise == 5:
        print(f'Q1. k = {k}, MSE = {mean_square_error(p.predict(test_X[0].to_numpy()), test_y.to_numpy())}') # res: 4, 27.20268809478639
    elif noise == 0:
        print(f'Q4. k = {k}, MSE = {mean_square_error(p.predict(test_X[0].to_numpy()), test_y.to_numpy())}') 
    else:
        print(f'Q5. k = {k}, MSE = {mean_square_error(p.predict(test_X[0].to_numpy()), test_y.to_numpy())}') 


def validation(estimator, l, train_X, train_y, test_X ,test_y, range):
    idx = np.argmin(np.array(l))
    k =  np.array(range)[idx]
    p = estimator(k)
    p.fit(train_X, train_y)
    if type(estimator) is RidgeRegression:print('Ridge:')
    else: print('lasso:')
    print(f'Best k = {k}, MSE = {mean_square_error(p.predict(test_X), test_y)}')




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train, X_test, y_test = X[:n_samples, ], y[:n_samples], X[n_samples:, ], y[n_samples:]


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    range_ridge = np.linspace(0.001, 10, n_evaluations)
    train_error_ridge, val_error_ridge = [], []
    for i in range_ridge: 
        estimator = RidgeRegression(i)
        tmp_train_ridge, tmp_val_ridge = cross_validate(estimator, X_train, y_train, mean_square_error, 5)
        train_error_ridge.append(tmp_train_ridge)
        val_error_ridge.append(tmp_val_ridge)

    
    range_lasso = np.linspace(0.001, 5, n_evaluations)
    train_error_lasso, val_error_lasso = [], []
    for i in range_lasso: 
        estimator_lasso = Lasso(i)
        tmp_train_lasso, tmp_val_lasso = cross_validate(estimator_lasso, X_train, y_train, mean_square_error, 5)
        train_error_lasso.append(tmp_train_lasso)
        val_error_lasso.append(tmp_val_lasso)

    
    fig3 = go.Figure()
    fig4 = go.Figure()
    fig3.add_trace(go.Scatter(name='trainError', x=range_ridge, y=train_error_ridge, mode='markers+lines', marker_color='red'))
    fig3.add_trace(go.Scatter(name='validationError', x=range_ridge, y=val_error_ridge, mode='markers+lines', marker_color='green'))
    fig4.add_trace(go.Scatter(name='trainError', x=range_lasso, y=train_error_lasso, mode='markers+lines', marker_color='red'))
    fig4.add_trace(go.Scatter(name='validationError', x=range_lasso, y=val_error_lasso, mode='markers+lines', marker_color='green'))

    fig3.update_layout(title=r"$\text{ }\text{training and Validation errors (average)}$", xaxis_title=r"$\text{Polynomial Degree}$",yaxis_title=r"$\text{errorValue}$", height=500, width=750)
    fig4.update_layout(title=r"$\text{ }\text{training and Validation errors (average)}$", xaxis_title=r"$\text{Polynomial Degree}$",yaxis_title=r"$\text{errorValue}$", height=500, width=750)

    fig3.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q7_ridge.png')
    fig4.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex5/photos/Q7_lasso.png')


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    validation(RidgeRegression, val_error_ridge, X_train, y_train, X_test, y_test, range_ridge)##ridge
    validation(Lasso, val_error_lasso, X_train, y_train, X_test, y_test, range_lasso)##lasso

    p = LinearRegression()
    p.fit(X_train, y_train)
    print(f'linear regression = {mean_square_error(p.predict(X_test), y_test)}')


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
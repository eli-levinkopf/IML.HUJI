
import sys
sys.path.append('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI')
from sklearn.metrics import average_precision_score
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import NoReturn
from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils import split_train_test
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)

    irrelevant_features = ['id', 'lat', 'long', 'yr_renovated']
    positive_features = ['price', 'sqft_living',
                         'sqft_lot', 'yr_built', 'sqft_living15', 'sqft_lot15']
    non_negative_features = ['bedrooms', 'bathrooms',
                             'floors', 'sqft_above', 'sqft_basement']
    rated_features = {"waterfront": [0, 1], "view": range(
        5), 'condition': range(1, 6), 'grade': range(1, 13)}

    # drop duplicte simpals and simpals that have missing features
    df = pd.DataFrame(data=data).drop_duplicates().dropna()
    # binery feature 'renovated_in_last_25Y'
    df['renovated_in_last_25Y'] = 2015 - df['yr_renovated'] <= 25
    #  drop irrelevant features
    for feature in irrelevant_features:
        df = df.drop(feature, axis=1)
    # drop non-positive features that most to be positive
    for feature in positive_features:
        df = df[df[feature] > 0]
    # drop negative features that most to be non-negative
    for feature in non_negative_features:
        df = df[df[feature] >= 0]
    # drop rated features with out of range rate
    for feature in rated_features.keys():
        df = df[df[feature].isin(rated_features[feature])]

    # Convert categorical features 'date', 'yr_built' and 'zipcode to indicator features
    df['date'] = pd.Series(df['date'].T).str.slice(stop=6)
    df = pd.get_dummies(df, prefix='date', columns=['date']).astype(int)
    df = pd.get_dummies(df, prefix='yr_built', columns=['yr_built']).astype(int)
    df = pd.get_dummies(df, prefix='zipcode', columns=['zipcode']).astype(int)

    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        if 'date' in feature:
            break
        rho = np.cov(X[feature], y) [0, 1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(x=X[feature], y=y, labels={'x':f'{feature}', 'y':'Response'},
                         title=rf'$\textbf{{Correlation between {feature} and response (price)}}\\\mathbf{{\rho}}\textbf{{ = {rho}}}$')

        fig.write_image(f'{output_path}correlation_for_%s.png' % feature)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data('datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    design_matrix, response_vector = df.drop('price', axis=1), df['price']
    # feature_evaluation(design_matrix, response_vector, '/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/') #plots

    # Question 3 - Split samples into training - and testing sets.
    train_X, train_Y, test_X, test_Y = split_train_test(design_matrix, response_vector, .75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    linear_regression = LinearRegression(include_intercept=True)
    range = np.arange(10, 101)
    res = np.empty(shape=(0,0))
    std = np.empty(shape=(0,0))
    for i in range:
        mse_array = []
        for j in np.arange(10):
            train_X_tmp, train_Y_tmp, test_X_tmp, test_Y_tmp = split_train_test(train_X, train_Y, i/100)
            linear_regression._fit(train_X_tmp, train_Y_tmp)
            loss = linear_regression._loss(test_X, test_Y)
            mse_array.append(loss)
        res = np.append(res, np.mean(mse_array))
        std = np.append(std, np.std(mse_array))

    fig = go.Figure(layout=go.Layout(title_text='mean loss as a function of percentage of training set'
                                    ,xaxis={"title": 'percentage'}, yaxis={"title": 'MSE of test set'}))
    fig.add_traces(go.Scatter(x=range,y=res, mode="markers+lines", name='Mean Of MSE', marker=dict(color="blue", opacity=.7)))
    fig.add_traces(go.Scatter(x=range, y=res - 2*std, fill=None, mode='lines', line=dict(color="lightgrey"), showlegend=False))
    fig.add_traces(go.Scatter(x=range, y=res + 2*std, fill='tonexty', mode='lines', line=dict(color="lightgrey"), showlegend=False))
    # fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/mse.png')
    fig.write_image('mse.png')

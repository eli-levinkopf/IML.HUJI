# TODO: Remove before submission
import sys
sys.path.append('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI')
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
from calendar import month_name 


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'])
    df = pd.DataFrame(data=data).drop_duplicates().dropna()
    df['dayofyear'] = df['Date'].dt.dayofyear
    df = df.drop('Date', axis=1)
    df = df[df['Temp'] > -20]
    df = df[df['City'].isin(['Capetown', 'Amsterdam', 'Tel Aviv', 'Amman'])]

    # df.to_csv('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI/datasets/Preprocess_City_Temperature.csv')
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI/datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    data_from_isreal = df[df['Country'] == 'Israel']
    dayofyear = np.arange(1, 366)
    fig1 = go.Figure(layout=go.Layout(title_text='Temperature as a function of day of year in Israel',
                                      xaxis={'title':'day of year'}, yaxis={'title':'Temperature'}))
    for year in set(data_from_isreal['Year']):
        fig1.add_traces(go.Scatter(x=dayofyear ,y=data_from_isreal[data_from_isreal['Year'] == year]['Temp'],
                                   type='scatter', mode='markers', name=f'{year}'))
    # fig1 = px.scatter(data_from_isreal, y='Temp', color='Year', color_discrete_sequence=get_colors(20))
    # fig1.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/TempInIsreal.png')

    dataframe = df.groupby('Month').agg('std')
    fig2 = px.bar(dataframe, y='Temp', pattern_shape=list(month_name)[1:], title='std of temperature in Israel per month')
    fig2.update_yaxes(title_text='std of Temperature')
    # fig2.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/TempPerMonthInIsreal.png')

    # Question 3 - Exploring differences between countries
    data_group_by_country = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp':['mean', 'std']})
    fig3 = px.line(x=data_group_by_country['Month'], y=data_group_by_country[('Temp', 'mean')], 
                   color=data_group_by_country['Country'], labels={'x':'Month', 'y':'Mean Temperature', 'color':'Country'},
                   error_y=data_group_by_country[('Temp', 'std')], title='Mean Temperature per Month')
    # fig3.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/MeanTemperaturePerMonth.png')


    # Question 4 - Fitting model for different values of `k`
    design_matrix, response_vector = df.drop(['City', 'Temp', 'Month', 'Day'], axis=1), df['Temp']
    # design_matrix.insert(0, 'include_intercept', 1, True) 

    # design_matrix['Country'] = design_matrix['Country'].str.replace('South Africa', '1')
    # design_matrix['Country'] = design_matrix['Country'].str.replace('The Netherlands', '2')
    # design_matrix['Country'] = design_matrix['Country'].str.replace('Israel', '3')
    # design_matrix['Country'] = design_matrix['Country'].str.replace('Jordan', '4')

    countries = {'South Africa':1, 'The Netherlands':2,'Israel':3, 'Jordan':4}
    for country in countries:
        design_matrix['Country'] = design_matrix['Country'].str.replace(country, str(countries[country]))
    design_matrix['Country'] = design_matrix['Country'].astype(int)

    train_X, train_Y, test_X, test_Y = split_train_test(design_matrix, response_vector, .75)

    mse_array = np.empty(shape=(0,0))
    for k in np.arange(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit._fit(train_X, train_Y)
        mse_array = np.append(mse_array, np.round(poly_fit._loss(test_X, test_Y), decimals=2))
    print(mse_array)
        

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()


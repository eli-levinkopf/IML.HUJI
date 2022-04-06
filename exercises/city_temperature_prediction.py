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
    # fig1.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/TempInIsreal.png')
    fig1.write_image('TempInIsreal.png')

    dataframe = data_from_isreal.groupby('Month').agg('std')
    fig2 = px.bar(dataframe, x=list(month_name)[1:], y='Temp', title='std of temperature in Israel per month')
    fig2.update_yaxes(title_text='std of Temperature')
    # fig2.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/TempPerMonthInIsreal.png')
    fig2.write_image('TempPerMonthInIsreal.png')

    # Question 3 - Exploring differences between countries
    data_group_by_country = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp':['mean', 'std']})
    fig3 = px.line(x=data_group_by_country['Month'], y=data_group_by_country[('Temp', 'mean')], 
                   color=data_group_by_country['Country'], labels={'x':'Month', 'y':'Mean Temperature', 'color':'Country'},
                   error_y=data_group_by_country[('Temp', 'std')], title='Mean Temperature per Month')
    # fig3.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/MeanTemperaturePerMonth.png')
    fig3.write_image('MeanTemperaturePerMonth.png')


    # Question 4 - Fitting model for different values of `k`
    design_matrix, response_vector = data_from_isreal.drop(['Country','Year', 'City', 'Month', 'Day', 'Temp'], axis=1), data_from_isreal['Temp']

    countries = {'South Africa':1, 'The Netherlands':2,'Israel':3, 'Jordan':4}
    train_X, train_Y, test_X, test_Y = split_train_test(design_matrix, response_vector, .75)

    ideal_k = 0
    min_error = np.inf
    test_error = np.empty(shape=(0,0))
    for k in np.arange(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit._fit(train_X.dayofyear, train_Y)
        error = np.round(poly_fit._loss(test_X.dayofyear, test_Y), decimals=2)
        if error < min_error:
            min_error = error
            ideal_k = k
        test_error = np.append(test_error, error)

    print(f'test_error: {test_error}')
    fig4 = px.bar(x=np.arange(1, 11), y=test_error, pattern_shape=np.arange(1, 11), title='Test error as a function of k',
                  labels={'x':'k', 'y':'test error'})
    # fig4.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/TestError.png')        
    fig4.write_image('TestError.png')        


    # Question 5 - Evaluating fitted model on different countries
    errors = np.empty(shape=(0,0))
    design_matrix, response_vector = df.drop(['Year', 'City', 'Month', 'Day', 'Temp'], axis=1), df['Temp']
    isreal_model = PolynomialFitting(ideal_k)
    isreal_model._fit(data_from_isreal.dayofyear, data_from_isreal['Temp'])
    for country in countries:
        country_data = df[df['Country'] == country]
        errors = np.append(errors, np.round(isreal_model._loss(country_data.dayofyear, country_data['Temp']), 3))
    
    fig5 = px.bar(x=countries.keys() ,y=errors, title='Israel model error over each of the other countries', 
                  labels={'x':'Country', 'y':'Israel model error'})
    # fig5.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex2/plots/TestErrorVSIsraelModel.png')
    fig5.write_image('TestErrorVSIsraelModel.png')
    

    
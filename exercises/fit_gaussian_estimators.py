from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots




pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    UniGaussian = UnivariateGaussian()
    mu = 10.0
    var = 1.0
    numOfSamples = 1000
    X = np.random.normal(mu, var, numOfSamples)
    UniGaussian.fit(X)
    print(f"(expectation, variance) = ({UniGaussian.mu_}, {UniGaussian.var_})")


    # Question 2 - Empirically showing sample mean is consistent
    axis = np.linspace(0, 1000, 100)
    estimatedArray = np.zeros(100)
    for i in range(1, 101):
        UniGaussian.fit(X[:i * 10])
        estimatedArray[i - 1] = UniGaussian.mu_
    
    fig1 = make_subplots()
    fig1.add_traces([go.Scatter(x=axis, y=abs(estimatedArray - mu), mode='markers')])
    fig1.update_layout(height=700, width=700, title_text="Question 2")
    fig1.update_xaxes(title_text="number of samples")
    fig1.update_yaxes(title_text="mu absolute distance")
    fig1.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    X.sort()
    pdf = UniGaussian.pdf(X)
    UniGaussian.fit(X)
    fig2 = make_subplots()
    fig2.add_traces([go.Scatter(x=X, y=pdf, mode='markers')])
    fig2.update_layout(height=700, width=700, title_text="Question 3")
    fig2.update_xaxes(title_text="sample values")
    fig2.update_yaxes(title_text="PDF values")
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    MultiGaussian = MultivariateGaussian()
    mu = np.array([0., 0., 4., 0.]).T
    cov = np.array([[1., 0.2, 0., 0.5],
                      [0.2, 2., 0., 0.],
                      [0., 0., 1., 0.],
                      [0.5, 0., 0., 1.]])
    numOfSamples = 1000
    X = np.random.multivariate_normal(mu, cov, numOfSamples)
    MultiGaussian.fit(X)
    print(f" estimated expectation:\n{MultiGaussian.mu_}")
    print(f"covariance matrix:\n{MultiGaussian.cov_}")


    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(-10, 10, 200)
    heatmap_matrix = np.zeros((f1.shape[0], f1.shape[0]))

    max_value = float('-inf')
    max_value_f1, max_value_f3 = None, None

    for i in range(f1.shape[0]):
        for j in range(f3.shape[0]):
            currMu = np.array([f1[i], 0, f3[j], 0], dtype=float)
            logLikelihood = MultiGaussian.log_likelihood(currMu, cov, X)
            heatmap_matrix[i][j] = logLikelihood
            if logLikelihood > max_value:
                max_value_f1, max_value_f3 = f1[i], f3[j]
                max_value = logLikelihood

    fig3 = make_subplots()
    fig3.add_traces(go.Heatmap(x=f1, y=f3, z=heatmap_matrix))
    fig3.update_layout(height=700, width=700, title_text="Question 5")
    fig3.update_xaxes(title_text="f1 values")
    fig3.update_yaxes(title_text="f3 values")
    fig3.show()


    # Question 6 - Maximum likelihood
    print(f"(f1, f3) that achieved the maximum log-likelihood: ({round(max_value_f1, 3)}, {round(max_value_f3, 3)})")


if __name__ == '__main__':
    test_univariate_gaussian()
    test_multivariate_gaussian()


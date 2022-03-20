from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    UniGaussian = UnivariateGaussian()
    mu = 10.0
    var = 1.0
    sigma = var ** 0.5
    numOfSamples = 1000
    X = np.random.normal(mu, var, numOfSamples)
    UniGaussian.fit(X)
    print(tuple([UniGaussian.mu_, UniGaussian.var_]))

    # plot(X)

    # Question 2 - Empirically showing sample mean is consistent
    axis = np.linspace(0, 1000, 100)
    estimatedArray = np.zeros(100)
    for i in range(1, 101):
        UniGaussian.fit(X[:i * 10])
        estimatedArray[i - 1] = UniGaussian.mu_
    fig1, ax1 = plt.subplots()
    ax1.plot(axis, abs(estimatedArray - mu))
    ax1.set_xlabel("number of samples")
    ax1.set_ylabel("mu absolute distance")
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    X.sort()
    pdf = UniGaussian.pdf(X)
    UniGaussian.fit(X)
    fig2, ax2 = plt.subplots()
    ax2.scatter(X, pdf)
    ax2.set_xlabel("sample values")
    ax2.set_ylabel("PDF values")
    fig2.show()

    # print(UniGaussian.log_likelihood(mu, sigma, X))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    MultiGaussian = MultivariateGaussian()
    mu = np.array([0., 0., 4., 0.]).transpose()
    sigma = np.array([[1., 0.2, 0., 0.5],
                      [0.2, 2., 0., 0.],
                      [0., 0., 1., 0.],
                      [0.5, 0., 0., 1.]])
    numOfSamples = 1000
    X = np.random.multivariate_normal(mu, sigma, numOfSamples)
    MultiGaussian.fit(X)
    print(MultiGaussian.mu_)
    print(MultiGaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(-10, 10, 200)
    heatmap_matrix = np.zeros((f1.shape[0], f1.shape[0]))

    max_value = float('-inf')
    max_value_f1, max_value_f3 = None, None

    for i in range(f1.shape[0]):
        for j in range(f3.shape[0]):
            currMu = np.array([f1[i], 0, f3[j], 0], dtype=float)
            logLikelihood = MultiGaussian.log_likelihood(currMu, sigma, X)
            heatmap_matrix[i][j] = logLikelihood
            if logLikelihood > max_value:
                max_value_f1, max_value_f3 = f1[i], f3[j]
                max_value = logLikelihood

    # fig3, ax3 = plt.subplots()
    # # sns.heatmap(heatmap_matrix)
    # ax3.imshow(heatmap_matrix, origin='lower', cmap='cubehelix',
    #            aspect='auto', interpolation='nearest',
    #            extent=[-10, 10, -10, 10])
    # ax3.set_xlabel("f1 values")
    # ax3.set_ylabel("f2 values")
    # fig3.show()

    fig3 = go.Figure(go.Heatmap(x=f1, y=f3, z=heatmap_matrix),
                     layout=go.Layout(height=500, width=500))
    fig3.update_xaxes(title_text="f1 values")
    fig3.update_yaxes(title_text="f3 values")
    fig3.show()

    # Question 6 - Maximum likelihood
    print(tuple([max_value_f1, max_value_f3]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
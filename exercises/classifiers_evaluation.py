# TODO: Remove before submission
import sys
sys.path.append('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI')
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
import math


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load('/Users/elilevinkopf/Documents/Ex22B/IML/IML.HUJI/datasets/' + filename)
    return data[:, :2], data[:, 2].astype(int)


def loss_callback(fit: Perceptron, x: np.ndarray, y: int):
    fit.training_loss_.append(fit.loss(x, y))


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = Perceptron(callback=loss_callback).fit(X, y).training_loss_

        # Plot figure
        title = ' training loss as function of training iterations'
        fig = go.Figure(layout=go.Layout(title_text=rf'$\textbf{{{n}{title}}}$',
                                    xaxis={'title':r'$\text{number of iterations}$'},
                                    yaxis={'title':r'$\text{loss values}$'}))
        fig.add_traces(go.Scatter(x=list(range(len(losses))), y=losses, mode='lines', showlegend=False))
        fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex3/plots/'+n+'.png')

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = math.atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_pred = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        import pandas as pd
        
        df = pd.DataFrame(np.column_stack((X[:, 0], X[:, 1])), columns=['x axis', 'y axis'])
        lda_accuracy = accuracy(y_true=y, y_pred=lda_pred)
        gaussian_accuracy = accuracy(y_true=y, y_pred=gnb_pred)
        print(gaussian_accuracy, lda_accuracy)
        title = [rf'$\textbf{{DataSet: {f}}}\\\textbf{{{model} accurancy: {accurancy}}}$' for
                 (model, accurancy) in [('LDA', lda_accuracy), ('GNB', gaussian_accuracy)]]
        fig = make_subplots(subplot_titles=title, rows=1, cols=2)

        fig.add_trace(go.Scatter(x=df['x axis'], y=df['y axis'], mode='markers',
         marker=dict(color=lda_pred, symbol=lda_pred, line=dict(color=y, width=1))), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['x axis'], y=df['y axis'], mode='markers',
         marker=dict(color=gnb_pred, symbol=lda_pred, line=dict(color=y, width=1))), row=1, col=2)

        fig.add_trace(get_ellipse(lda.mu_[:, 0], lda.cov_), row=1, col=1)
        fig.add_trace(get_ellipse(lda.mu_[:, 1], lda.cov_), row=1, col=1)
        fig.add_trace(get_ellipse(lda.mu_[:, 2], lda.cov_), row=1, col=1)
        fig.add_trace(go.Scatter(x=[lda.mu_[:, 0][0]], y=[lda.mu_[:, 0][1]], 
         mode='markers', showlegend=False,marker=dict(color='black', symbol='x')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[lda.mu_[:, 1][0]], y=[lda.mu_[:, 1][1]],
         mode='markers', showlegend=False,marker=dict(color='black', symbol='x')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[lda.mu_[:, 2][0]], y=[lda.mu_[:, 2][1]],
         mode='markers', showlegend=False,marker=dict(color='black', symbol='x')), row=1, col=1)

        fig.add_trace(get_ellipse(gnb.mu_[0], np.diag(np.array(gnb.vars_[0:1, :2])[0])), row=1, col=2)
        fig.add_trace(get_ellipse(gnb.mu_[1], np.diag(np.array(gnb.vars_[1:2, :2])[0])), row=1, col=2)
        fig.add_trace(get_ellipse(gnb.mu_[2], np.diag(np.array(gnb.vars_[2:3, :2])[0])), row=1, col=2)
        fig.add_trace(go.Scatter(x=[gnb.mu_[0][0]], y=[gnb.mu_[0][1]],
         mode='markers', showlegend=False,marker=dict(color='black', symbol='x')), row=1, col=2)
        fig.add_trace(go.Scatter(x=[gnb.mu_[1][0]], y=[gnb.mu_[1][1]],
         mode='markers', showlegend=False,marker=dict(color='black', symbol='x')), row=1, col=2)
        fig.add_trace(go.Scatter(x=[gnb.mu_[2][0]], y=[gnb.mu_[2][1]],
         mode='markers', showlegend=False,marker=dict(color='black', symbol='x')), row=1, col=2)

        fig.update_layout(width=1000, height=700, showlegend=False)

        fig.show()
        # fig.write_image('/Users/elilevinkopf/Documents/Ex22B/IML/ex3/plots/'+f+'.png')
        


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

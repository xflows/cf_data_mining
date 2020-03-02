#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . scikit_classifier import ScikitLearnClassifier

__author__ = 'daleksovski'


def regression_tree(**kwargs):
    from sklearn import tree
    tree = tree.DecisionTreeRegressor(**kwargs)
    return ScikitLearnClassifier(tree)


def lasso_LARS(alpha=1.0):
    """L1-regularized least squares linear classifier trained with Least Angle Regression. alpha=constant that multiplies the penalty term, default 1.0

    :param alpha: Constant that multiplies the penalty term.
    :return: a LassoLars object
    """

    from sklearn.linear_model import LassoLars
    clf = LassoLars(alpha=alpha)

    return ScikitLearnClassifier(clf)


def sgd_regressor():
    """Linear classifier fitted by minimizing a regularized empirical loss with Stochastic Gradient Descent

    :return: a SGDRegressor object
    """

    from sklearn.linear_model import SGDRegressor
    clf = SGDRegressor()
    return ScikitLearnClassifier(clf)


def ard_regression(num_iter=300):
    """Bayesian Automated Relevance Determination regression.

    :param num_iter: maximum number of iterations, default 300
    :return: a ARDRegression object
    """

    from sklearn.linear_model import ARDRegression
    clf = ARDRegression(n_iter=num_iter)
    return ScikitLearnClassifier(clf)


def ridge_regression():
    """ L2-regularized least squares linear classifier

    :return: a Ridge object
    """

    from sklearn.linear_model import Ridge
    clf = Ridge()
    return ScikitLearnClassifier(clf)

def elastic_net_regression():
    """L1+L2-regularized least squares linear classifier trained using Coordinate Descent

    :return: an ElasticNet object
    """

    from sklearn.linear_model import ElasticNet
    clf = ElasticNet()
    return ScikitLearnClassifier(clf)


def svr():
    """Epsilon-Support Vector Regression, using the RBF kernel

    :return: a SVR object
    """
    from sklearn.svm import SVR
    clf = SVR()
    return ScikitLearnClassifier(clf)

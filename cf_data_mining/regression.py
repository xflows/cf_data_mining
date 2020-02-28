#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . scikit_classifier import ScikitLearnClassifier

__author__ = 'daleksovski'

def regression_tree(max_features="auto", max_depth=None):
    """Decision tree for regression problems

    :param featureIn: The number of features to consider when looking for the best split: If int, then consider max_features features at each split; If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split
    :param max_depth: The maximum depth of the tree
    :return: a DecisionTreeRegressor object
    """

    from sklearn import tree

    clf = tree.DecisionTreeRegressor(max_features=max_features, max_depth=max_depth)

    return ScikitLearnClassifier(clf)


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

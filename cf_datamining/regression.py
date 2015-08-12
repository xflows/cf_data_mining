#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'darkoa'

def decisionTreeRegressor(maxFeaturesIn="auto", depthIn=None):
    """Decision tree for regression problems

    :param featureIn: The number of features to consider when looking for the best split: If int, then consider max_features features at each split; If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split
    :param depthIn: The maximum depth of the tree
    :return: a DecisionTreeRegressor object
    """

    from sklearn import tree

    clf = tree.DecisionTreeRegressor(max_features=maxFeaturesIn, max_depth=depthIn)

    return clf


def lassoLARS(alphaIn=1.0):
    """L1-regularized least squares linear model trained with Least Angle Regression. alpha=constant that multiplies the penalty term, default 1.0

    :param alphaIn: Constant that multiplies the penalty term.
    :return: a LassoLars object
    """

    from sklearn.linear_model import LassoLars
    clf = LassoLars(alpha=alphaIn)

    return clf


def sgdRegressor():
    """Linear model fitted by minimizing a regularized empirical loss with Stochastic Gradient Descent

    :return: a SGDRegressor object
    """

    from sklearn.linear_model import SGDRegressor
    clf = SGDRegressor()
    return clf


def ardRegression(numIterationsIn=300):
    """Bayesian Automated Relevance Determination regression.

    :param numIterationsIn: maximum number of iterations, default 300
    :return: a ARDRegression object
    """

    from sklearn.linear_model import ARDRegression
    clf = ARDRegression(n_iter=numIterationsIn)
    return clf


def ridge():
    """ L2-regularized least squares linear model

    :return: a Ridge object
    """

    from sklearn.linear_model import Ridge
    clf = Ridge()
    return clf

def elasticNet():
    """L1+L2-regularized least squares linear model trained using Coordinate Descent

    :return: an ElasticNet object
    """

    from sklearn.linear_model import ElasticNet
    clf = ElasticNet()
    return clf


def svr():
    """Epsilon-Support Vector Regression, using the RBF kernel

    :return: a SVR object
    """
    from sklearn.svm import SVR
    clf = SVR()
    return clf

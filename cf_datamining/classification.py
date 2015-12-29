#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scikit_classifier import ScikitLearnClassifier

__author__ = 'darkoa'

def naive_bayes():
    """Naive Bayes algorithm for classification

    :return: a ScikitLearner object, containing a GaussianNB learner
    """
    from sklearn.naive_bayes import GaussianNB
    gaussianNBLearner = GaussianNB()

    # return gaussianNBLearner
    return ScikitLearnClassifier(gaussianNBLearner)


def SVC(penaltyIn=1.0, kernelIn="rbf", degIn=3):
    """Support Vector Machines with kernels based on libsvm

    :param penaltyIn: Penalty parameter C of the error term. float
    :param kernelIn: Specifies the kernel type to be used in the algorithm. string
    :param degIn: Degree of the polynomial kernel function (‘poly’). int
    :return: a SVC object.
    """

    from sklearn.svm import SVC
    # clf = SVC(C=float(input_dict["penaltyIn"]), kernel=str(input_dict["kernelIn"]), degree=int(input_dict["degIn"]))
    clf = SVC(C=float(penaltyIn), kernel=str(kernelIn), degree=int(degIn))
    return ScikitLearnClassifier(clf)


def k_nearest_neighbors(numNeighbIn=5, weithgsIn='uniform', algIn='auto'):
    """k-Nearest Neighbors classifier based on the ball tree datastructure for low dimensional data and brute force search for high dimensional data

    :param numNeighbIn: Number of neighbors to use by default for k_neighbors queries.
    :param weithgsIn: Weight function used in prediction.
    :param algIn: Algorithm used to compute the nearest neighbors.
    :return: a KNeighborsClassifier object.
    """

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=int(numNeighbIn), weights=str(weithgsIn), algorithm=str(algIn))
    return ScikitLearnClassifier(knn)


def logistic_regression(penaltyIn="l1", cIn=1.0):
    """Logistic regression classifier.

    :param penaltyIn: the penalty, (string) used to specify the norm used in the penalization. ‘l1’ or ‘l2’.
    :param cIn: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    :return: a LogisticRegression object
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty=str(penaltyIn), C=float(cIn))
    return ScikitLearnClassifier(clf)


def linear_SVC(cIn=1.0, lossIn="l2", penaltyIn="l2", multiClassIn="ovr"):
    """Support Vector Regression, implemented in terms of liblinear

    :param cIn: Penalty parameter C of the error term. float, default=1.0
    :param lossIn: Specifies the loss function. string, ‘l1’ or ‘l2’, default=’l2’
    :param penaltyIn: Specifies the norm used in the penalization. string, ‘l1’ or ‘l2’, default 'l2'
    :param multiClassIn:
    :return: a LinearSVC object.
    """
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=float(cIn),loss=str(lossIn), penalty=str(penaltyIn), multi_class=str(multiClassIn), dual=True )

    return ScikitLearnClassifier(clf)


def J48(maxFeaturesIn="auto", depthIn=None):
    """ Creates a J48 decision tree classifier

    :param maxFeaturesIn: The number of features to consider when looking for the best split
    :param depthIn: The maximum depth of the tree
    :return: a DecisionTreeClassifier object
    """

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(max_features=maxFeaturesIn, max_depth=depthIn)

    return ScikitLearnClassifier(clf)

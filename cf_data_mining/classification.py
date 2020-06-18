#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . scikit_classifier import ScikitLearnClassifier

__author__ = 'daleksovski'

def naive_bayes():
    """Naive Bayes algorithm for classification

    :return: a ScikitLearner object, containing a GaussianNB learner
    """
    from sklearn.naive_bayes import GaussianNB
    gaussian_NB_learner = GaussianNB()

    # return gaussianNBLearner
    return ScikitLearnClassifier(gaussian_NB_learner)


def SVC(C=1.0, kernel="rbf", degree=3):
    """Support Vector Machines with kernels based on libsvm

    :param penalty: Penalty parameter C of the error term. float
    :param kernel: Specifies the kernel type to be used in the algorithm. string
    :param degree: Degree of the polynomial kernel function (‘poly’). int
    :return: a SVC object.
    """

    from sklearn.svm import SVC
    # clf = SVC(C=float(input_dict["penalty"]), kernel=str(input_dict["kernel"]), degree=int(input_dict["degree"]))
    clf = SVC(C=float(C), kernel=str(kernel), degree=int(degree))
    return ScikitLearnClassifier(clf)


def k_nearest_neighbors(num_neighbors=5, weithgs='uniform', alg='auto'):
    """k-Nearest Neighbors classifier based on the ball tree datastructure for low dimensional data and brute force search for high dimensional data

    :param num_neighbors: Number of neighbors to use by default for k_neighbors queries.
    :param weithgs: Weight function used in prediction.
    :param alg: Algorithm used to compute the nearest neighbors.
    :return: a KNeighborsClassifier object.
    """

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=int(num_neighbors), weights=str(weithgs), algorithm=str(alg))
    return ScikitLearnClassifier(knn)


def logistic_regression(penalty="l1", c=1.0):
    """Logistic regression classifier.

    :param penalty: the penalty, (string) used to specify the norm used in the penalization. ‘l1’ or ‘l2’.
    :param c: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    :return: a LogisticRegression object
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty=str(penalty), C=float(c))
    return ScikitLearnClassifier(clf)


def linear_SVC(c=1.0, loss="l2", penalty="l2", multi_class="ovr"):
    """Support Vector Regression, implemented in terms of liblinear

    :param c: Penalty parameter C of the error term. float, default=1.0
    :param loss: Specifies the loss function. string, ‘l1’ or ‘l2’, default=’l2’
    :param penalty: Specifies the norm used in the penalization. string, ‘l1’ or ‘l2’, default 'l2'
    :param multi_class:
    :return: a LinearSVC object.
    """
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=float(c),loss=str(loss), penalty=str(penalty), multi_class=str(multi_class), dual=True)

    return ScikitLearnClassifier(clf)


# def J48(max_features="auto", depth=None):
#     """ Creates a J48 decision tree classifier
#
#     :param max_features: The number of features to consider when looking for the best split
#     :param depth: The maximum depth of the tree
#     :return: a DecisionTreeClassifier object
#     """
#
#     from sklearn import tree
#     clf = tree.DecisionTreeClassifier(max_features=max_features, max_depth=depth)
#
#     return ScikitLearnClassifier(clf)


def decision_tree(**kwargs):
    from sklearn import tree
    tree = tree.DecisionTreeClassifier(**kwargs)
    return ScikitLearnClassifier(tree)


def dummy_classifier(strategy):
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy)
    return ScikitLearnClassifier(dummy)

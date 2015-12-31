#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

def build_classifier(classifier, data):
    '''Builds a classifier

    :param classifier: a Classifier object
    :param data: a SciKit dataset structure
    '''

    # generic, for all DataMining libraries
    # -------------------------------------
    classifier.build_classifier(data)



def apply_classifier(classifier, data):
    '''Applies a classifier on a dataset, and gets predictions

    :param classifier: a classifier
    :param data: a SciKit dataset
    :return: the input data containing a key targetPredicted with the classifier predictions
    '''

    # generic, for all DataMining libraries
    # -------------------------------------
    new_data = classifier.apply_classifier(data)

    return new_data


def helper_extract_true_values_and_predictions(data):
    y_true = data["target"]
    y_pred = data["targetPredicted"]
    return (y_true, y_pred)

def accuracy_score(data):
    '''Calculates accuracy of a classification classifier

    :param data: a SciKit dataset, containing key targetPredicted
    :return: accuracy, float
    '''
    y_true, y_pred = helper_extract_true_values_and_predictions(data)
    from sklearn.metrics import accuracy_score
    result_acc = accuracy_score( y_true, y_pred )

    return result_acc


def mse(data):
    '''Calculates mean_squared_error (MSE) of a regression classifier

    :param data: a SciKit dataset, containing key targetPredicted
    :return: MSE, float
    '''
    from sklearn.metrics import mean_squared_error
    y_true, y_pred = helper_extract_true_values_and_predictions(data)

    result_mse = mean_squared_error(y_true, y_pred)


    return result_mse

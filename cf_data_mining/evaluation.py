#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

from sklearn import metrics

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

def calculate_classification_statistics(dataset):

    # Old format:
    # 'Expected true and predicted labels for each fold, but failed.' +
    # 'If you wish to provide labels for each fold separately it should look like: ' +
    # '[[y_true_1, y_predicted_1], [y_true_2, y_predicted_2], ...]')

    labels = [[],[]]
    for i in range(0,len(dataset.target)):
        labels[0].append(dataset['target'][i])
        labels[1].append(dataset['targetPredicted'][i])

    # Check if we have true and predicted labels for each fold
    if labels and type(labels[0][0]) == list:
        try:
            # Flatten
            y_true, y_pred = [], []
            for fold_labels in labels:
                y_true.extend(fold_labels[0])
                y_pred.extend(fold_labels[1])
            labels = [y_true, y_pred]
        except:
            raise Exception('Expected true and predicted labels for each fold, but failed.' +
                            'If you wish to provide labels for each fold separately it should look like: ' +
                            '[[y_true_1, y_predicted_1], [y_true_2, y_predicted_2], ...]')
    if len(labels) != 2:
        raise Exception('Wrong input structure, this widget accepts labels in the form: [y_true, y_pred]')

    y_true, y_pred = labels

    classes = set()
    classes.update(y_true + y_pred)
    classes = sorted(list(classes))

    # Assign integers to classes
    class_to_int = {}
    for i, cls_label in enumerate(classes):
        class_to_int[cls_label] = i

    y_true = [class_to_int[lbl] for lbl in y_true]
    y_pred = [class_to_int[lbl] for lbl in y_pred]

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    if len(classes) == 2:
        auc = metrics.auc_score(y_true, y_pred)
    else:
        auc = 'AUC for multiclass problems requires class probabilities'

    return accuracy, precision, recall, f1, auc, confusion_matrix

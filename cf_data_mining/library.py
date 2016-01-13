#!/usr/bin/env python
# -*- coding: utf-8 -*- 
__author__ = 'daleksovski'

import numpy as np
import classification as c, regression as r
import evaluation as e, utilities as ut
import dataset

# -------------------
#   CLASSIFICATION
# -------------------

def naive_bayes(input_dict):
    """ Naive Bayes algorithm for classification """
    from sklearn.naive_bayes import GaussianNB
    y_pred = GaussianNB()
    output_dict = {}
    output_dict['bayes_out'] = c.naive_bayes()
    return output_dict


def k_nearest_neighbors(input_dict):
    """k-Nearest Neighbors classifier based on the ball tree data structure
    for low dimensional data and brute force search for high dimensional data"""

    knn = c.k_nearest_neighbors(input_dict['numNeib'], input_dict['wgIn'], input_dict['algIn'])
    output_dict = {}
    output_dict['KNN_out'] = knn
    return output_dict


def logistic_regression(input_dict):
    '''Logistic regression classifier.'''
    output_dict = {}
    output_dict['LR_out'] = c.logistic_regression(input_dict["pen_in"], input_dict["c_in"])
    return output_dict


def support_vector_machines_classification(input_dict):
    """Support Vector Machines with kernels based on libsvm"""
    output_dict = {}
    output_dict['SVC_out'] = c.SVC(input_dict["penalty_in"], input_dict["kernel_in"], input_dict["deg_in"])
    return output_dict


def support_vector_machines_classification_using_liblinear(input_dict):
    """ Support Vector Regression, without kernels, based on liblinear """

    clf = c.linear_SVC(c=float(input_dict["c_in"]), loss=input_dict["loss_in"], penalty=input_dict["penalty_in"],
                       multi_class=input_dict["multi_class_in"])
    output_dict = {}
    output_dict['SVC_out'] = clf
    return output_dict


def decision_tree(input_dict):
    """ Creates a J48 decision tree classifier """

    # parse input and determine its type
    try:
        # return int or float
        maxFeatures = float(input_dict["max_features_in"]) if '.' in input_dict["max_features_in"] else int(
                input_dict["max_features_in"])
    except ValueError:
        maxFeatures = input_dict["max_features_in"]  # return string

    clf = c.J48(max_features=maxFeatures, depth=int(input_dict["depth_in"]))

    output_dict = {}
    output_dict['tree_out'] = clf
    return output_dict


# -------------------
#   REGRESSION
# -------------------

def ard_regression(input_dict):
    """ Bayesian Automated Relevance Determination regression. n_iter=maximum number of iterations, default 300 """

    clf = r.ard_regression(num_iter=int(input_dict["n_iter"]))

    output_dict = {}
    output_dict['out'] = clf
    return output_dict


def regression_tree(input_dict):
    # parse input and determine its type
    try:
        maxFeatures = float(input_dict["maxFeaturesIn"]) if '.' in input_dict["maxFeaturesIn"] else int(
                input_dict["maxFeaturesIn"])  # return int or float
    except ValueError:
        maxFeatures = input_dict["maxFeaturesIn"]  # return string

    clf = r.regression_tree(maxFeatures, int(input_dict["depthIn"]))

    output_dict = {}
    output_dict['tree_out'] = clf
    return output_dict


def lasso_LARS(input_dict):
    """ L1-regularized least squares linear classifier trained with Least Angle Regression. alpha=constant
    that multiplies the penalty term, default 1.0 """

    clf = r.lasso_LARS(alpha=float(input_dict["alpha"]))

    output_dict = {}
    output_dict['out'] = clf
    return output_dict


def sgd_regressor(input_dict):
    """ Linear classifier fitted by minimizing a regularized empirical loss with Stochastic Gradient Descent. """

    clf = r.sgd_regressor()
    output_dict = {}
    output_dict['out'] = clf
    return output_dict


def ridge_regression(input_dict):
    """ L2-regularized least squares linear classifier """
    clf = r.ridge_regression()
    output_dict = {}
    output_dict['out'] = clf
    return output_dict


def elastic_net_regression(input_dict):
    """ L1+L2-regularized least squares linear classifier trained using Coordinate Descent. """

    clf = r.elastic_net_regression()
    output_dict = {}
    output_dict['out'] = clf
    return output_dict


def support_vector_regression(input_dict):
    """ Epsilon-Support Vector Regression, using the RBF kernel. """

    clf = r.svr()
    output_dict = {}
    output_dict['out'] = clf
    return output_dict


# ----------------------------
#   EVALUATION
# ----------------------------

def classification_accuracy(input_dict):
    """
    Calculates classification accuracy.
    Expects a SciKit dataset structure on input, with the field 'targetPredicted'
    """

    acc = e.accuracy_score(input_dict["data"])
    return {'ca': acc}


def regression_mse(input_dict):
    """
    Calculates mean_squared_error (MSE)
    """
    mse = e.mse(input_dict["data"])
    return {'mse': mse}


def build_classifier(input_dict):
    """ Builds a classifier """

    e.build_classifier(input_dict['learner'], input_dict["instances"])

    output_dict = {'classifier': input_dict['learner']}
    return output_dict


def apply_classifier(input_dict):
    """ Applies a built classifier on a dataset """

    new_data = e.apply_classifier(input_dict['classifier'], input_dict['data'])

    output_dict = {'classes': new_data}
    return output_dict


def classification_statistics(input_dict):

    accuracy, precision, recall, f1, auc, confusion_matrix = e.calculate_classification_statistics(input_dict['dataset'])

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc, 'confusion_matrix': confusion_matrix}



# ----------------------------
#   UTILITIES
# ----------------------------

def print_model(input_dict):
    """Outputs textual information about a model"""
    classifier = input_dict['model']
    output_dict = {}
    output_dict['model_as_string'] = classifier.print_classifier()
    return output_dict


def display_classifier(input_dict):
    """Displays a classifier/model"""
    return {}


def export_dataset_to_csv(input_dict):
    """ Exports a dataset to a CSV file """
    return {}


def load_UCI_dataset(input_dict):
    """ Loads a UCI dataset """
    dataset = ut.load_UCI_dataset(input_dict['dsIn'])

    output_dict = {}
    output_dict['dtsOut'] = dataset
    return output_dict


def display_decision_tree(input_dict):
    """ Displays a decision tree """
    from sklearn import tree
    from StringIO import StringIO
    classifier = input_dict['classifier'].classifier
    out = StringIO()
    out = tree.export_graphviz(classifier, out_file=out)
    import StringIO
    from os import system

    tree.export_graphviz(classifier, out_file="decisionTreeJ48-scikit.dot")  # dotfile)

    # CORRECT SO THAT IMAGE IS GOING TO BE SAVED IN THE CORRECT DIRECTORY
    system("dot -Tpng decisionTreeJ48-scikit.dot -o workflows/static/decision_tree.png")

    return {}


def import_dataset_from_csv(input_dict):
    """ Imports CSV file, and creates a Scikit dataset. """

    # the target value must be in the last column of the CSV file
    output_dict = {}
    # this code converts data from the csv file into scikit learn dataset and returns it as a tuple
    import numpy

    # my_data = numpy.genfromtxt(input_dict['fileIn'], delimiter=',')
    from StringIO import StringIO
    my_data = numpy.genfromtxt(StringIO(input_dict['fileIn']), delimiter=',')

    num_samples, num_attributes = np.shape(my_data)
    num_targets = 1

    data = np.empty((num_samples, num_attributes - num_targets))
    target = np.empty((num_samples,))

    for i in range(0, num_samples):
        data[i] = np.asarray(my_data[i][:-1])
        target[i] = np.asarray(my_data[i][-1])

    from sklearn.datasets import base as ds
    dataset = ds.Bunch(data=data,
                       target=target,
                       feature_names=[],
                       DESCR="",
                       target_names="")

    output_dict['dataset'] = dataset
    return output_dict


def split_dataset_randomly(input_dict):
    """ Randomly splits a given dataset into a train and test dataset."""

    inst = input_dict['data']
    test_size = 1 - float(input_dict["p"])

    # train test split
    from sklearn.cross_validation import train_test_split
    data_train, data_test, target_train, target_test = train_test_split(
            inst['data'],
            inst['target'],
            test_size=test_size,
            random_state=1)

    from sklearn.datasets import base as ds

    if dataset.is_target_nominal(inst):
        a_train = ds.Bunch(data=data_train,
                           target=target_train,
                           feature_names=inst.feature_names,
                           DESCR=inst.DESCR,
                           target_names=inst.target_names)

        a_test = ds.Bunch(data=data_test,
                          target=target_test,
                          feature_names=inst.feature_names,
                          DESCR=inst.DESCR,
                          target_names=inst.target_names)
    else:
        a_train = ds.Bunch(data=data_train,
                           target=target_train,
                           feature_names=inst.feature_names,
                           DESCR=inst.DESCR)

        a_test = ds.Bunch(data=data_test,
                          target=target_test,
                          feature_names=inst.feature_names,
                          DESCR=inst.DESCR)

    if inst.has_key("feature_value_names"):
        a_train["feature_value_names"] = inst.feature_value_names
        a_test["feature_value_names"] = inst.feature_value_names

    return {'train_data': a_train, 'test_data': a_test}


def select_data(input_dict):
    return input_dict


def select_data_post(postdata, input_dict, output_dict):
    import json

    data = input_dict['data']
    data_compl = np.hstack([np.matrix(data.data), np.transpose(np.matrix(data.target))])
    data_compl = np.array(data_compl)

    conditions = json.loads(str(postdata['conditions'][0]))

    print postdata['conditions'][0]

    attr_names = data.feature_names
    attr_names.append('class')

    for cond in conditions['conditions']:
        for or_cond in cond['condition']:

            attr_ind = list(attr_names).index(str(or_cond['attr']))

            op = str(or_cond['operator'])
            val = list(or_cond['values'])

            if op in ['outside', 'between', 'is defined']:
                # TODO Fix
                raise NotImplementedError("Not implemented")

            if op == 'in':
                # nominal attrubite
                att_vals = data.feature_value_names[attr_ind]
                inds = [att_vals.index(v) for v in val]

                res = np.empty([0, np.shape(data_compl)[1]])
                for ind in inds:
                    matrix_part = data_compl[data_compl[:, attr_ind] == ind, :]
                    res = np.vstack((res, matrix_part))
                data_compl = res

            if op in ['>=', '<=', '>', '<', '=']:
                val = [float(el) for el in val]
                if op == '>=':
                    data_compl = data_compl[data_compl[:, attr_ind] >= val[0], :]
                if op == '<=':
                    data_compl = data_compl[data_compl[:, attr_ind] <= val[0], :]
                if op == '<':
                    data_compl = data_compl[data_compl[:, attr_ind] < val[0], :]
                if op == '>':
                    data_compl = data_compl[data_compl[:, attr_ind] > val[0], :]
                if op == '=':
                    data_compl = data_compl[data_compl[:, attr_ind] == val[0], :]

    output_dict['data']['data'] = data_compl[:, 0:-1]
    output_dict['data']['target'] = data_compl[:, -1]

    return output_dict


def display_dataset(input_dict):
    return {}


def display_clustering_table_form(input_dict):
    return {}

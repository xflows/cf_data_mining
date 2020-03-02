#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

import numpy as np
import arff
from . import classification as c
from . import regression as r
from . import evaluation as e
from . import utilities as ut
from . import dataset

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
    """ Creates a Decision tree classifier """

    criterion = input_dict['criterion']
    splitter = input_dict['splitter']
    max_depth = input_dict['max_depth']
    min_samples_leaf = input_dict['min_samples_leaf']

    max_depth = max_depth.strip()
    if max_depth == '':
        max_depth = None
    else:
        max_depth = int(max_depth)

    min_samples_leaf = min_samples_leaf.strip()
    if min_samples_leaf == '':
        min_samples_leaf = 1
    else:
        min_samples_leaf = int(min_samples_leaf)

    clf = c.decision_tree(criterion=criterion,
                          splitter=splitter,
                          max_depth=max_depth,
                          min_samples_leaf=min_samples_leaf)
    return {'tree': clf}


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
    """ Creates a Decision tree classifier """

    criterion = input_dict['criterion']
    splitter = input_dict['splitter']
    max_depth = input_dict['max_depth']
    min_samples_leaf = input_dict['min_samples_leaf']

    max_depth = max_depth.strip()
    if max_depth == '':
        max_depth = None
    else:
        max_depth = int(max_depth)

    min_samples_leaf = min_samples_leaf.strip()
    if min_samples_leaf == '':
        min_samples_leaf = 1
    else:
        min_samples_leaf = int(min_samples_leaf)

    reg = r.regression_tree(criterion=criterion,
                            splitter=splitter,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf)
    return {'tree': reg}



def regression_treeOLD(input_dict):
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
    # """ Displays a decision tree """
    # from sklearn import tree
    # import tempfile
    # import subprocess
    # import os
    #
    # from mothra.settings import MEDIA_ROOT
    # from workflows.helpers import ensure_dir
    #
    # dataset = input_dict.get('dataset')
    # if dataset is not None:
    #     feature_names = dataset.get('feature_names')
    #     class_names = dataset.get('target_names')
    # else:
    #     feature_names = class_names = None
    #
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.dot') as fp:
    #     tree.export_graphviz(input_dict['classifier'].classifier,
    #                          out_file=fp.name,
    #                          feature_names=feature_names,
    #                          class_names=class_names)
    #     fp.flush()
    #     path, fname = os.path.split(fp.name)
    #     base, ext = os.path.splitext(fname)
    #     pngfile = os.path.join(MEDIA_ROOT, 'sklearn', base + '.png')
    #     print(pngfile)
    #     ensure_dir(pngfile)
    #     subprocess.Popen(['dot', '-Tpng', '-o', pngfile, fp.name])
    #
    # return {'pngfile': pngfile}

    return {}

def import_dataset_from_csv(input_dict):
    """ Imports CSV file, and creates a Scikit dataset. """

    cindex = input_dict.get('target_index')
    if isinstance(cindex, str):
        cindex = cindex.strip()
        if cindex == '':
            cindex = None
        elif str(cindex).lower() == 'last':
            cindex = -1
        elif str(cindex).lower() == 'first':
            cindex = 0
        else:
            cindex = int(cindex)

    output_dict = {}
    import numpy as np
    import csv

    # data = np.genfromtxt(input_dict['fileIn'], delimiter=',')

    with open(input_dict['fileIn']) as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        has_header = csv.Sniffer().sniff(sample)
        reader = csv.reader(csvfile, dialect)
        rows = [line for line in reader]

    if has_header:
        feature_names = rows[0]
        del rows[0]
    else:
        feature_names = None

    # separate target column if required
    if cindex is not None:
        y = []
        X = []
        for row in rows:
            y.append(row[cindex])
            del(row[cindex])
            X.append(row)
        del feature_names[cindex]
    else:
        X = rows
        y = None
    # X = np.asarray(X, dtype=np.float)

    # build column arrays and encode discrete features where required
    from sklearn.preprocessing import OrdinalEncoder
    X = np.array(X)
    enc = OrdinalEncoder()
    data = []
    for idx in range(0, X.shape[1]):
        try:
            col = np.array(X[:, idx], dtype=np.float)
        except ValueError:
            col = np.array(X[:, idx], dtype=np.unicode)
            col = col.reshape(-1, 1)  # encoder works on column vectors
            enc.fit(col)
            col = enc.transform(col).reshape(-1)
        data.append(col)
    X = np.array(data).T

    # encode discrete target
    if cindex is not None:
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        target_names = le.classes_
    else:
        target_names = None

    from sklearn.datasets import base as ds
    dataset = ds.Bunch(data=X,
                       target=y,
                       feature_names=feature_names,
                       DESCR="",
                       target_names="")

    output_dict['dataset'] = dataset
    return output_dict


def split_dataset_randomly(input_dict):
    """ Randomly splits a given dataset into a train and test dataset."""

    inst = input_dict['data']
    test_size = 1 - float(input_dict["p"])
    seed = int(input_dict['seed'])
    is_stratified = True if input_dict['strat'] == 'true' else False

    # train test split
    from sklearn.model_selection import train_test_split
    data_train, data_test, target_train, target_test = train_test_split(inst['data'],
                                                                        inst['target'],
                                                                        test_size=test_size,
                                                                        random_state=seed,
                                                                        stratify=inst['target'] if is_stratified else None)

    from sklearn.datasets import base as ds

    a_train = ds.Bunch(data=data_train,
                       target=target_train,
                       feature_names=inst.feature_names,
                       DESCR=inst.DESCR)

    a_test = ds.Bunch(data=data_test,
                      target=target_test,
                      feature_names=inst.feature_names,
                      DESCR=inst.DESCR)

    return {'train_data': a_train, 'test_data': a_test}


def select_data(input_dict):
    return input_dict


def select_data_post(postdata, input_dict, output_dict):
    import json

    data = input_dict['data']
    data_compl = np.hstack([np.matrix(data.data), np.transpose(np.matrix(data.target))])
    data_compl = np.array(data_compl)

    conditions = json.loads(str(postdata['conditions'][0]))

    print((postdata['conditions'][0]))

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



def import_dataset_from_arff(input_dict):
    if not input_dict['arff_file']:
        raise ValueError('Input file is required!')
    arff_file = input_dict['arff_file']

    cindex = input_dict.get('target_index')
    if isinstance(cindex, str):
        cindex = cindex.strip()
        if cindex == '':
            cindex = None
        elif str(cindex).lower() == 'last':
            cindex = -1
        elif str(cindex).lower() == 'first':
            cindex = 0
        else:
            cindex = int(cindex)

    with open(arff_file) as fp:
        afd = arff.load(fp)

    # get feature types and names
    atypes = []
    feature_names = []
    for aname, atype in afd['attributes']:
        feature_names.append(aname)
        if isinstance(atype, list):
            atypes.append(np.unicode)
        else:
            atype = atype.lower()
            if atype in ['numeric', 'real', 'float', 'double']:
                atypes.append(np.float)
            elif atype in ['date', 'string', 'text']:
                atypes.append(np.unicode)
            elif atype in ['integer', 'int']:
                atypes.append(np.int)
            else:
                atypes.append(np.unicode)

    # separate target column if required
    if cindex is not None:
        y = []
        X = []
        for row in afd['data']:
            y.append(row[cindex])
            del(row[cindex])
            X.append(row)
        y_type = atypes[cindex]
        del(feature_names[cindex])
        del(atypes[cindex])
    else:
        X = afd['data']
        y = None

    # build column arrays and encode discrete features where required
    from sklearn.preprocessing import OrdinalEncoder
    X = np.array(X)
    enc = OrdinalEncoder()
    data = []
    for idx in range(0, X.shape[1]):
        col = np.array(X[:, idx], dtype=atypes[idx])
        if atypes[idx] in [np.unicode, np.int]:
            col = col.reshape(-1, 1)  # encoder works on column vectors
            enc.fit(col)
            col = enc.transform(col).reshape(-1)
            data.append(col)
        elif atypes[idx] in [np.float]:
            data.append(col)
        else:
            raise ValueError('Unsupported type {}'.format(str(atypes[idx])))
    X = np.array(data).T

    # encode discrete target
    if cindex is not None:
        if y_type in [np.unicode, np.int]:
            from sklearn import preprocessing
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
            target_names = le.classes_
        else:
            target_names = None
    else:
        target_names = None

    from sklearn.datasets import base as ds
    dataset = ds.Bunch(data=X,
                       target=y,
                       feature_names=feature_names,
                       target_names=target_names)
    return {'dataset': dataset}

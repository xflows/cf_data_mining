#!/usr/bin/env python
# -*- coding: utf-8 -*- 

__author__ = 'darkoa'

import numpy as np
import classification as c, regression as r, unsupervised as u
import evaluation as e, utilities as ut

# -------------------
#   CLASSIFICATION
# -------------------

def naive_bayes(input_dict):
    """ Naive Bayes algorithm for classification """
    from sklearn.naive_bayes import GaussianNB 
    y_pred = GaussianNB()
    output_dict={}
    output_dict['bayesout'] = c.naiveBayes()
    return output_dict

def support_vector_machines_classification(input_dict):
    """Support Vector Machines with kernels based on libsvm"""
    output_dict={}
    output_dict['SVCout'] = c.SVC( input_dict["penaltyIn"], input_dict["kernelIn"], input_dict["degIn"])
    return output_dict

def k_nearest_neighbors(input_dict):
    """k-Nearest Neighbors classifier based on the ball tree datastructure for low dimensional data and brute force search for high dimensional data"""

    knn = c.kNearestNeighbors(input_dict['numNeib'], input_dict['wgIn'], input_dict['algIn'] )
    output_dict={}
    output_dict['KNNout'] = knn
    return output_dict


def logistic_regression(input_dict):
    '''Logistic regression classifier.'''
    output_dict={}
    output_dict['LRout'] = c.logisticRegression(input_dict["penIn"], input_dict["cIn"])
    return output_dict

def linear_svc(input_dict):
    """ Support Vector Regression, without kernels, based on liblinear """

    clf = c.linearSVC(cIn=float(input_dict["cIn"]),lossIn=input_dict["lossIn"],penaltyIn=input_dict["penaltyIn"], multiClassIn=input_dict["multiClassIn"])
    output_dict={}
    output_dict['SVCout'] = clf
    return output_dict


def j48(input_dict):
    """ Creates a J48 decision tree classifier """

    #parse input and determine its type
    try:
        maxFeatures= float(input_dict["maxFeaturesIn"]) if '.' in input_dict["maxFeaturesIn"] else int(input_dict["maxFeaturesIn"]) #return int or float
    except ValueError:
        maxFeatures= input_dict["maxFeaturesIn"] #return string

    clf = c.J48(maxFeaturesIn=maxFeatures, depthIn=int(input_dict["depthIn"]))

    output_dict={}
    output_dict['treeOut'] = clf
    return output_dict



# -------------------
#   REGRESSION
# -------------------

def regression_tree(input_dict):
    #parse input and determine its type
    try:
        maxFeatures= float(input_dict["maxFeaturesIn"]) if '.' in input_dict["maxFeaturesIn"] else int(input_dict["maxFeaturesIn"]) #return int or float
    except ValueError:
        maxFeatures= input_dict["maxFeaturesIn"] #return string

    clf = r.decisionTreeRegressor(maxFeatures, int( input_dict["depthIn"] ))

    output_dict={}
    output_dict['treeOut'] = clf
    return output_dict


def lasso_LARS(input_dict):
    """ L1-regularized least squares linear model trained with Least Angle Regression. alpha=constant that multiplies the penalty term, default 1.0 """

    clf = r.lassoLARS(alphaIn=float(input_dict["alpha"]))

    output_dict={}
    output_dict['out'] = clf
    return output_dict

def sgd_regressor(input_dict):
    """ Linear model fitted by minimizing a regularized empirical loss with Stochastic Gradient Descent. """

    clf = r.sgdRegressor()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def ard_regression(input_dict):
    """ Bayesian Automated Relevance Determination regression. n_iter=maximum number of iterations, default 300 """

    clf = r.ardRegression(int(input_dict["n_iter"]))
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def ridge_regression(input_dict):
    """ L2-regularized least squares linear model """
    clf = r.ridge()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def elastic_net_regression(input_dict):
    """ L1+L2-regularized least squares linear model trained using Coordinate Descent. """

    clf = r.elasticNet()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def support_vector_regression(input_dict):
    """ Epsilon-Support Vector Regression, using the RBF kernel. """

    clf = r.svr()
    output_dict={}
    output_dict['out'] = clf
    return output_dict



# --------------------
#   UNSUPERVISED
# --------------------

def k_means(input_dict):
    """k-Means clustering"""

    kMeansClusterCenters, clusteredData =  u.kMeans(input_dict['instances'], input_dict['k'])
    return {'clusterCenters':kMeansClusterCenters, 'clusteredData':clusteredData}


def aglomerative_clustering(input_dict):
    """  Hierarchical Agglomerative Clustering, using the Ward linkage and euclidean metric. The parameter k (num.clusters) needs to be set, default value 3. """
    clusteredData = u.aglomerativeClustering(input_dict['instances'], input_dict['k'])
    return {'clusteredData':clusteredData}



# ----------------------------
#   EVALUATION
# ----------------------------

def classification_accuracy(input_dict):
    """
    Calculates classification accuracy.
    Expects a SciKit dataset structure on input, with the field 'targetPredicted'
    """

    acc = e.accuracyScore(input_dict["data"])
    return { 'ca':acc }


def regression_mse(input_dict):
    """
    Calculates mean_squared_error (MSE)
    """
    mse = e.mse(input_dict["data"])
    return {'mse': mse}



def build_classifier(input_dict):
    """ Builds a classifier """

    clf = e.buildClassifier(input_dict['learner'], input_dict["instances"])

    output_dict = {'classifier': clf}
    return output_dict

def apply_classifier(input_dict):
    """ Applies a built classifier on a dataset """

    new_data = e.applyClassifier(input_dict['classifier'], input_dict['data'])

    output_dict = {'classes':new_data}
    return output_dict



# ----------------------------
#   UTILITIES
# ----------------------------


def export_dataset_to_CSV(input_dict):
    """ Exports a dataset to a CSV file """
    return {}


def load_UCI_dataset(input_dict):
    """ Loads a UCI dataset """

    dataset = ut.load_UCI_dataset(input_dict['dsIn'])

    output_dict = {}
    output_dict['dtsOut'] = dataset#(dataset.data, dataset.target)
    return output_dict




def display_decision_tree(input_dict):
    """ Displays a decision tree """
    from sklearn import tree
    from StringIO import StringIO
    out = StringIO()
    out = tree.export_graphviz(input_dict['classifier'], out_file=out)
    import StringIO, pydot 
    from os import system
    dot_data = StringIO.StringIO() 

    tree.export_graphviz(input_dict['classifier'], out_file="decisionTreeJ48-scikit.dot") #dotfile)
    # dotfile.close()
    system("dot -Tpng decisionTreeJ48-scikit.dot -o workflows/static/decisionTree-scikit.png") #CORRECT SO THAT IMAGE IS GOING TO BE SAVED IN THE CORRECT DIRECTORY
    return {}


def import_dataset_from_csv(input_dict):
    """ Imports CSV file, and creates a Scikit dataset. """
    # the target value must be in the last column of the CSV file
    output_dict={}
    # this code converts data from the csv file into scikit learn dataset and returns it as a tuple
    import numpy

    # my_data = numpy.genfromtxt(input_dict['fileIn'], delimiter=',')
    from StringIO import StringIO
    my_data = numpy.genfromtxt(StringIO(input_dict['fileIn']), delimiter=',')

    num_samples, num_attributes = np.shape(my_data)
    num_targets = 1

    data = np.empty( (num_samples, num_attributes - num_targets) )
    target = np.empty((num_samples,))

    for i in range(0,num_samples):
        data[i] = np.asarray(my_data[i][:-1])
        target[i] = np.asarray(my_data[i][-1])

    from sklearn.datasets import base as ds
    dataset = ds.Bunch(data=data,
                 target=target,
                 feature_names=[],
                 DESCR="",
                 target_names="")

    output_dict['scikitDataset'] =  dataset
    return output_dict # returns a touple consiting of n_samples x n_features numpy array X and an array of length n_samples containing the targets y

def split_dataset(input_dict):
    """ Randomly splits a given dataset into a train and test dataset."""

    inst = input_dict['data']
    test_size = 1 - float( input_dict["p"] )

    # train test split
    from sklearn.cross_validation import train_test_split
    data_train, data_test, target_train, target_test = train_test_split(
        inst['data'],
        inst['target'],
        test_size=test_size, 
        random_state=1)

    from sklearn.datasets import base as ds
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

    if inst.has_key("feature_value_names"):
        a_train["feature_value_names"] = inst.feature_value_names
        a_test["feature_value_names"] = inst.feature_value_names

    return {'train_data':a_train, 'test_data':a_test}


def select_data(input_dict):
    return input_dict

def select_data_post(postdata, input_dict, output_dict):
    import json

    data = input_dict['data']
    data_compl  = np.hstack( [np.matrix(data.data), np.transpose( np.matrix(data.target)) ])
    data_compl  = np.array( data_compl )

    conditions = json.loads(str(postdata['conditions'][0]))

    print postdata['conditions'][0]

    for cond in conditions['conditions']:
        if cond['condition'][0]['operator'] in ["is defined", "sis defined"]:
            print "***** "
        else:
            for or_cond in cond['condition']:
                #arr = data.feature_names
                attrInd = list(data.feature_names).index( str(or_cond['attr']) )

                op = str(or_cond['operator'])
                val= list(or_cond['values'])
                val= [float(el) for el in val]
                if op=='>=':
                    # my_inds = np.where( data_mat[:,attrInd] >= val[0] )
                    data_compl = data_compl[ data_compl[:,attrInd] >= val[0], :  ]
                if op=='<=':
                    data_compl = data_compl[ data_compl[:,attrInd] <= val[0], :  ]
                if op=='<':
                    data_compl = data_compl[ data_compl[:,attrInd] < val[0], :  ]
                if op=='>':
                    data_compl = data_compl[ data_compl[:,attrInd] > val[0], :  ]
                if op=='=':
                    data_compl = data_compl[ data_compl[:,attrInd] == val[0], :  ]
                if op in ['outside', 'between', 'is defined'] :
                    raise NotImplementedError


    output_dict['data']['data']     = data_compl[:, 0:-1]
    output_dict['data']['target']   = data_compl[ :, -1 ]

    return output_dict

    #return {'data': data, 'dummy':5}

# ===================================================

def display_dataset(input_dict):
    return {}


def display_clustering_table_form(input_dict):
    return {}


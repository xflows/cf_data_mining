#!/usr/bin/env python
# -*- coding: utf-8 -*- 

__author__ = 'darkoa'

import numpy as np
import classification as c, regression as r, unsupervised as u


# -------------------
#   CLASSIFICATION
# -------------------

def scikitAlgorithms_naiveBayes(input_dict):
    """ Naive Bayes algorithm for classification """
    from sklearn.naive_bayes import GaussianNB 
    y_pred = GaussianNB()
    output_dict={}
    output_dict['bayesout'] = c.naiveBayes()
    return output_dict

def scikitAlgorithms_SVC(input_dict):
    """Support Vector Machines with kernels based on libsvm"""
    output_dict={}
    output_dict['SVCout'] = c.SVC( input_dict["penaltyIn"], input_dict["kernelIn"], input_dict["degIn"])
    return output_dict

def scikitAlgorithms_kNearestNeighbors(input_dict):
    """k-Nearest Neighbors classifier based on the ball tree datastructure for low dimensional data and brute force search for high dimensional data"""

    knn = c.kNearestNeighbors(input_dict['numNeib'], input_dict['wgIn'], input_dict['algIn'] )
    output_dict={}
    output_dict['KNNout'] = knn
    return output_dict

def scikitAlgorithms_logisticRegression(input_dict):
    '''Logistic regression classifier.'''
    output_dict={}
    output_dict['LRout'] = c.logisticRegression(input_dict["penIn"], input_dict["cIn"])
    return output_dict

def scikitAlgorithms_linearSVC(input_dict):
    """ Support Vector Regression, without kernels, based on liblinear """

    clf = c.linearSVC(cIn=float(input_dict["cIn"]),lossIn=input_dict["lossIn"],penaltyIn=input_dict["penaltyIn"], multiClassIn=input_dict["multiClassIn"])
    output_dict={}
    output_dict['SVCout'] = clf
    return output_dict

def scikitAlgorithms_kNearestNeighbors(input_dict):
    """ k-Nearest Neighbors classifier based on the ball tree datastructure for low dimensional data and brute force search for high dimensional data """
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=int(input_dict['numNeib']), weights=input_dict['wgIn'], algorithm=input_dict['algIn'])
    output_dict={}
    output_dict['KNNout'] = knn
    return output_dict

def scikitAlgorithms_J48(input_dict):
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

def scikitAlgorithms_DecisionTreeRegressor(input_dict):
    #parse input and determine its type
    try:
        maxFeatures= float(input_dict["maxFeaturesIn"]) if '.' in input_dict["maxFeaturesIn"] else int(input_dict["maxFeaturesIn"]) #return int or float
    except ValueError:
        maxFeatures= input_dict["maxFeaturesIn"] #return string

    clf = r.decisionTreeRegressor(maxFeatures, int( input_dict["depthIn"] ))

    output_dict={}
    output_dict['treeOut'] = clf
    return output_dict


def scikitAlgorithms_LassoLARS(input_dict):
    """ L1-regularized least squares linear model trained with Least Angle Regression. alpha=constant that multiplies the penalty term, default 1.0 """

    clf = r.lassoLARS(alphaIn=float(input_dict["alpha"]))

    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_SGDRegressor(input_dict):
    """ Linear model fitted by minimizing a regularized empirical loss with Stochastic Gradient Descent. """

    clf = r.sgdRegressor()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_ARDRegression(input_dict):
    """ Bayesian Automated Relevance Determination regression. n_iter=maximum number of iterations, default 300 """

    clf = r.ardRegression(int(input_dict["n_iter"]))
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_Ridge(input_dict):
    """ L2-regularized least squares linear model """
    clf = r.ridge()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_ElasticNet(input_dict):
    """ L1+L2-regularized least squares linear model trained using Coordinate Descent. """

    clf = r.elasticNet()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_SVR(input_dict):
    """ Epsilon-Support Vector Regression, using the RBF kernel. """

    clf = r.svr()
    output_dict={}
    output_dict['out'] = clf
    return output_dict



# --------------------
#   UNSUPERVISED
# --------------------

def scikitAlgorithms_kMeans(input_dict):
    """k-Means clustering"""

    kMeansClusterCenters, clusteredData =  u.kMeans(input_dict['instances'], input_dict['k'])
    return {'clusterCenters':kMeansClusterCenters, 'clusteredData':clusteredData}


def scikitAlgorithms_AglomerativeClustering(input_dict):
    """  Hierarchical Agglomerative Clustering, using the Ward linkage and euclidean metric. The parameter k (num.clusters) needs to be set, default value 3. """
    clusteredData = u.aglomerativeClustering(input_dict['instances'], input_dict['k'])
    return {'clusteredData':clusteredData}


# ----------------------------
#   UTILITIES and EVALUATION
# ----------------------------

def scikitAlgorithms_buildClassifier(input_dict):
    """ Builds a classifier """

    learner = input_dict['learner']
    data = input_dict['instances']
    n_sample = data["data"]
    n_feature = data["target"]
    # print " --" + str(n_sample)
    # print "---" + str(n_feature)

    classifier = learner.fit(n_sample, n_feature) #.predict(n_sample)

    output_dict = {'classifier': classifier}
    return output_dict

def scikitAlgorithms_applyClassifier(input_dict):
    """ Applies a built classifier on a dataset """
    classifier = input_dict['classifier']
    data = input_dict['data']
    data["targetPredicted"] = classifier.predict(data["data"])

    # new_data = (data["data"], classifier.predict(data["data"]))
   
    output_dict = {'classes':data}
    return output_dict

def helper_extractTrueValuesAndPredictions(input_dict):
    y_true = input_dict["data"]["target"]
    y_pred = input_dict["data"]["targetPredicted"]
    return (y_true, y_pred)


def scikitAlgorithms_accuracyScore(input_dict):
    """ 
    Calculates classification accuracy.
    Expects a SciKit dataset structure on input, with the field 'targetPredicted'
    """
    y_true, y_pred = helper_extractTrueValuesAndPredictions(input_dict)
    from sklearn.metrics import accuracy_score
    resAcc = accuracy_score( y_true, y_pred )

    return { 'ca':resAcc }

def scikitAlgorithms_MSE(input_dict):
    """
    Calculates mean_squared_error (MSE)
    """
    from sklearn.metrics import mean_squared_error
    y_true, y_pred = helper_extractTrueValuesAndPredictions(input_dict)
    # print str( y_true )    print str( y_pred )
    resMSE = mean_squared_error(y_true, y_pred)
    # print str( resMSE )
    return {'mse': resMSE}



def scikitAlgorithms_displayDecisionTree(input_dict):
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


def scikitAlgorithms_UCIDataset(input_dict):
    """ Loads a UCI dataset """
    from sklearn import datasets
    allDSets = {"iris":datasets.load_iris(), "boston":datasets.load_boston(), "diabetes":datasets.load_diabetes(), " linnerud":datasets.load_linnerud()}
    dataset = allDSets[input_dict['dsIn']]
    output_dict = {}
    output_dict['dtsOut'] = dataset#(dataset.data, dataset.target)
    return output_dict



def scikitAlgorithms_scikitDatasetToCSV(input_dict):
    """ Exports a SciKit dataset to a CSV file """

    output_dict={}
    dataset= input_dict['scikitDataset']

    import numpy
    csv=[]
    count=0
    for i,sample in enumerate(dataset.data):
        csv.append(numpy.append(sample,dataset.target[i])) #join n_sample and n_feature array

    numpy.savetxt("foo.csv", csv, fmt='%.6f', delimiter=",")
    output_dict['CSVout'] = csv
    return output_dict



def scikitAlgorithms_CSVtoNumpy(input_dict):
    """ Imports CSV file, and creates a Scikit dataset. """
    # the targer value must be in the last colum of the CSV file
    output_dict={}
    # this code converts data from the csv file into scikit learn dataset and returns it as a tuple
    import numpy
    my_data = numpy.genfromtxt(input_dict['fileIn'], delimiter=',')
    n_sample = []
    n_feature = []
    for x in my_data:
        n_feature.append(x[-1]) 
        n_sample.append(x[:-1])
    print n_sample
    print n_feature
    dataset = (n_sample, n_feature)
    output_dict['scikitDataset'] =  dataset
    return output_dict # returns a touple consiting of n_samples x n_features numpy array X and an array of length n_samples containing the targets y

def scikitAlgorithms_split_dataset(input_dict):
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
                 DESCR=inst.DESCR)

    a_test = ds.Bunch(data=data_test,
                 target=target_test,
                 feature_names=inst.feature_names,
                 DESCR=inst.DESCR)

    return {'train_data':a_train, 'test_data':a_test}


def scikitAlgorithms_select_data(input_dict):
    return input_dict

def scikitAlgorithms_select_data_post(postdata, input_dict, output_dict):
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


    output_dict['data']['data'] = data_compl[:, 0:-1]
    output_dict['data']['target'] =  data_compl[ :, -1 ]

    return output_dict

    #return {'data': data, 'dummy':5}

# ===================================================

def scikitAlgorithms_displayDS(input_dict):
    return {}



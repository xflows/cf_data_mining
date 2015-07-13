#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#
#   CLASSIFICATION
#

def scikitAlgorithms_naiveBayes(input_dict):
    """ Naive Bayes algorithm for classification """
    from sklearn.naive_bayes import GaussianNB 
    y_pred = GaussianNB()
    output_dict={}
    output_dict['bayesout'] = y_pred
    return output_dict

def scikitAlgorithms_SVC(input_dict):
    """
    Support Vector Machines with kernels based on libsvm
    """
    from sklearn.svm import SVC
    clf = SVC(C=float(input_dict["penaltyIn"]), kernel=str(input_dict["kernelIn"]), degree=int(input_dict["degIn"]))
    output_dict={}
    output_dict['SVCout'] = clf
    return output_dict

def scikitAlgorithms_kNearestNeighbors(input_dict):
    """ 
    k-Nearest Neighbors classifier based on the ball tree datastructure for low dimensional data and brute force search for high dimensional data
    """
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=int(input_dict['numNeib']), weights=input_dict['wgIn'], algorithm=input_dict['algIn'])
    output_dict={}
    output_dict['KNNout'] = knn
    return output_dict

def scikitAlgorithms_linearSVC(input_dict):
    """ Support Vector Regression, without kernels, based on liblinear """
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=float(input_dict["penaltyIn"]),loss=input_dict["lossIn"],penalty=input_dict["normIn"], multi_class=input_dict["classIn"])
    output_dict={}
    output_dict['SVCout'] = clf
    return output_dict

def scikitAlgorithms_kNearestNeighbors(input_dict):
    """ TBD """
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=int(input_dict['numNeib']), weights=input_dict['wgIn'], algorithm=input_dict['algIn'])
    output_dict={}
    output_dict['KNNout'] = knn
    return output_dict

def scikitAlgorithms_logisticRegression(input_dict):
    '''Logistic regression classifier.
    The parameters are:
    penalty : {string} Used to specify the norm used in the penalization. ‘l1’ or ‘l2’.
    C : {float} Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.'''
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty=str(input_dict["penIn"]), C=float(input_dict["cIn"]))
    output_dict={}
    output_dict['LRout'] = clf
    return output_dict

def scikitAlgorithms_J48(input_dict):
    """ Creates a J48 decision tree classifier """
    from sklearn import tree
    #parse input and determin its type
    try:
        featureValue= float(input_dict["featureIn"]) if '.' in input_dict["featureIn"] else int(input_dict["featureIn"]) #return int or float
    except ValueError:
        featureValue= input_dict["featureIn"] #return string
    clf = tree.DecisionTreeClassifier(max_features=featureValue, max_depth=int(input_dict["depthIn"]))
    output_dict={}
    output_dict['treeOut'] = clf
    return output_dict
#
#   REGRESSION
#

def scikitAlgorithms_DecisionTreeRegressor(input_dict):
    from sklearn import tree
    #parse input and determin its type
    try:
        featureValue= float(input_dict["featureIn"]) if '.' in input_dict["featureIn"] else int(input_dict["featureIn"]) #return int or float
    except ValueError:
        featureValue= input_dict["featureIn"] #return string
    clf = tree.DecisionTreeRegressor(max_features=featureValue, max_depth=int(input_dict["depthIn"]))

    print "scikitAlgorithms_DecisionTreeRegressor :: " + str(clf)

    output_dict={}
    output_dict['treeOut'] = clf
    return output_dict


def scikitAlgorithms_LassoLARS(input_dict):
    """ TBD """
    from sklearn.linear_model import LassoLars
    clf = LassoLars(alpha=float(input_dict["authIn"]))
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_SGDRegressor(input_dict):
    """ TBD """
    from sklearn.linear_model import SGDRegressor
    clf = SGDRegressor()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_ARDRegression(input_dict):
    """ TBD """
    from sklearn.linear_model import ARDRegression
    clf = ARDRegression(n_iter=int(input_dict["iterIn"]))
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_SVR(input_dict):
    """ TBD """
    from sklearn.svm import SVR 
    clf = SVR()
    output_dict={}
    output_dict['out'] = clf
    return output_dict


def scikitAlgorithms_Ridge(input_dict):
    """ L2-regularized least squares linear model """
    from sklearn.linear_model import Ridge
    clf = Ridge()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

def scikitAlgorithms_ElasticNet(input_dict):
    """ TBD """
    from sklearn.linear_model import ElasticNet
    clf = ElasticNet()
    output_dict={}
    output_dict['out'] = clf
    return output_dict

#
#   UTILITIES and EVALUATION
#

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
    dotfile = open("decisionTreeJ48-scikit.dot", 'w')    
    if dotfile != None:
        print( type(dotfile) )
        dotfile = tree.export_graphviz(input_dict['classifier'], out_file=dotfile)         
        # dotfile.close()
        system("dot -Tpng decisionTreeJ48-scikit.dot -o decisionTreeJ48-scikit.png") #CORRECT SO THAT IMAGE IS GOING TO BE SAVED IN THE CORRECT DIRECTORY
    else:
        ErrorOpeningFile

def scikitAlgorithms_UCIDataset(input_dict):
    """ Loads a UCI dataset """
    from sklearn import datasets
    allDSets = {"iris":datasets.load_iris(), "boston":datasets.load_boston(), "diabetes":datasets.load_diabetes(), " linnerud":datasets.load_linnerud()}
    dataset = allDSets[input_dict['dsIn']]
    output_dict = {}
    output_dict['dtsOut'] = dataset#(dataset.data, dataset.target)
    return output_dict



def scikitAlgorithms_scikitDatasetToCSV(input_dict):
    """ TBD """
    output_dict={}
    dataset= input_dict['scikitDataset']
    n_sample = dataset[0]
    n_feature = dataset[1]
    import numpy
    csv=[]
    count=0
    for sample in n_sample:
        csv.append(numpy.append(sample,n_feature[count])) #join n_sample and n_feature array
        count+=1
    #numpy.savetxt("foo.csv", csv, delimiter=",")
    output_dict['CSVout'] = csv
    return output_dict



def scikitAlgorithms_CSVtoNumpy(input_dict):
    """ TBD """
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


# ===================================================
# ===================================================

def scikitAlgorithms_displayDS(input_dict):
    return {}



def helperDisplayDS(data):
    """ Temporarily placed here """

    #get data to fill table
    info = data['data']
    print type(info)

    n_sample = info["data"]
    n_feature = info["target"]

    # join data in the right format
    import numpy
    csv=[]
    count=0
    for sample in n_sample:
        csv.append(numpy.append(sample,n_feature[count])) #join n_sample and n_feature array
        count+=1

    attrs = ["attribute" for i in range(len(n_sample[0]))] #name of attributes
    attrs = info.feature_names
    class_var = 'category'
    metas = '' 
    data_new = csv #fill table with data
    
    return {'attrs':attrs, 'metas':metas, 'data_new':data_new, 'class_var':class_var}

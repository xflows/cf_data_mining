__author__ = 'darkoa'

def buildClassifier(classifier, data):
    '''Builds a classifier

    :param classifier: a Classifier object
    :param data: a SciKit dataset structure
    '''

    # generic, for all DataMining libraries
    # -------------------------------------------
    classifier.buildClassifier(data)

    # specific, only for Scikit DataMining library
    # -------------------------------------------
    # n_sample = data["data"]
    # n_feature = data["target"]
    # classifier = classifier.fit(n_sample, n_feature) #.predict(n_sample)



def applyClassifier(classifier, data):
    '''Applies a classifier on a dataset, and gets predictions

    :param classifier: a classifier
    :param data: a SciKit dataset
    :return: the input data containing a key targetPredicted with the classifier predictions
    '''

    # generic, for all DataMining libraries
    # -------------------------------------------
    newData = classifier.applyClassifier(data)

    # specific, only for Scikit DataMining library
    # -------------------------------------------
    # data["targetPredicted"] = classifier.predict(data["data"])

    return newData


def helperExtractTrueValuesAndPredictions(data):
    y_true = data["target"]
    y_pred = data["targetPredicted"]
    return (y_true, y_pred)

def accuracyScore(data):
    '''Calculates accuracy of a classification classifier

    :param data: a SciKit dataset, containing key targetPredicted
    :return: accuracy, float
    '''
    y_true, y_pred = helperExtractTrueValuesAndPredictions(data)
    from sklearn.metrics import accuracy_score
    resAcc = accuracy_score( y_true, y_pred )

    return resAcc


def mse(data):
    '''Calculates mean_squared_error (MSE) of a regression classifier

    :param data: a SciKit dataset, containing key targetPredicted
    :return: MSE, float
    '''
    from sklearn.metrics import mean_squared_error
    y_true, y_pred = helperExtractTrueValuesAndPredictions(data)
    # print str( y_true )    print str( y_pred )
    resMSE = mean_squared_error(y_true, y_pred)
    # print str( resMSE )

    return resMSE

__author__ = 'darkoa'

def buildClassifier(learner, data):
    '''Builds a classifier

    :param learner: the learner provided
    :param data: a SciKit dataset structure
    :return: a classifier
    '''

    # learner = input_dict['learner']
    # data = input_dict['instances']
    n_sample = data["data"]
    n_feature = data["target"]
    # print " --" + str(n_sample)
    # print "---" + str(n_feature)

    classifier = learner.fit(n_sample, n_feature) #.predict(n_sample)

    return classifier


def applyClassifier(classifier, data):
    '''Applies a classifier on a dataset, and gets predictions

    :param classifier: a classifier
    :param data: a SciKit dataset
    :return: the input data containing a key targetPredicted with the model predictions
    '''

    data["targetPredicted"] = classifier.predict(data["data"])

    return data




def helperExtractTrueValuesAndPredictions(data):
    y_true = data["target"]
    y_pred = data["targetPredicted"]
    return (y_true, y_pred)

def accuracyScore(data):
    '''Calculates accuracy of a classification model

    :param data: a SciKit dataset, containing key targetPredicted
    :return: accuracy, float
    '''
    y_true, y_pred = helperExtractTrueValuesAndPredictions(data)
    from sklearn.metrics import accuracy_score
    resAcc = accuracy_score( y_true, y_pred )

    return resAcc


def mse(data):
    '''Calculates mean_squared_error (MSE) of a regression model

    :param data: a SciKit dataset, containing key targetPredicted
    :return: MSE, float
    '''
    from sklearn.metrics import mean_squared_error
    y_true, y_pred = helperExtractTrueValuesAndPredictions(data)
    # print str( y_true )    print str( y_pred )
    resMSE = mean_squared_error(y_true, y_pred)
    # print str( resMSE )

    return resMSE

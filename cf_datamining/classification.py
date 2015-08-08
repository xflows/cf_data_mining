__author__ = 'darkoa'

def naiveBayes():
    """Naive Bayes algorithm for classification

    :return: a GaussianNB object from sklearn.naive_bayes
    """
    from sklearn.naive_bayes import GaussianNB
    y_pred = GaussianNB()
    return y_pred


def SVC(penaltyIn, kernelIn, degIn):
    """Support Vector Machines with kernels based on libsvm

    :param penaltyIn: float
    :param kernelIn: string, possible values: ""
    :param degIn: int
    :return: a SVC object from sklearn.svm
    """

    from sklearn.svm import SVC
    # clf = SVC(C=float(input_dict["penaltyIn"]), kernel=str(input_dict["kernelIn"]), degree=int(input_dict["degIn"]))
    clf = SVC(C=float(penaltyIn), kernel=str(kernelIn), degree=int(degIn))
    return clf


def kNearestNeighbors(numNeighbIn, weithgsIn, algIn):
    """k-Nearest Neighbors classifier based on the ball tree datastructure for low dimensional data and brute force search for high dimensional data

    :param numNeighbIn:
    :param weithgsIn:
    :param algIn:
    :return: a KNeighborsClassifier object from sklearn.neighbors
    """

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=int(numNeighbIn), weights=weithgsIn, algorithm=algIn)
    return knn


def logisticRegression(penaltyIn, cIn):
    """Logistic regression classifier.

    :param penaltyIn: the penalty, (string) used to specify the norm used in the penalization. ‘l1’ or ‘l2’.
    :param cIn: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    :return: a LogisticRegression object
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty=str(penaltyIn), C=float(cIn))
    return clf

__author__ = 'darkoa'

class Classifier(object):
    """
    Classifer class: wraps a Learner and a Model
    Needed if there is a need to use one of the widgets: "Build classifier", "Apply classifier", "Print model", "Cross-validation", etc.
    """

    def buildClassifier(self, data):
        """Builds a classifier

        :param data: bunch
        """
        raise NotImplementedError

    def applyClassifier(self, data):
        """Applies a classifier on a dataset, and gets predictions

        :param data: bunch
        :return: bunch with targetPredicted
        """
        raise NotImplementedError

    def printClassifier(self):
        """Prints the model/classifier
        """
        raise NotImplementedError

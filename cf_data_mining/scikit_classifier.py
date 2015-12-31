#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

from classifier import Classifier


class ScikitLearnClassifier(Classifier):
    """
    ScikitLearnClassifier class: extends Classifier (a Learner and a Model)
    Needed if there is a need to use one of the widgets: "Build classifier", "Apply classifier", "Print model", "Cross-validation", etc.
    """

    def __init__(self, learner):
        self.learner = learner
        self.classifier = None

    def build_classifier(self, data):
        """Builds a scikit classifier

        :param data: bunch
        :return:
        """
        n_sample = data["data"]
        n_feature = data["target"]
        # print " --" + str(n_sample)
        # print "---" + str(n_feature)

        self.classifier = self.learner.fit(n_sample, n_feature)

    def apply_classifier(self, data):
        """Applies a scikit classifier on a dataset, and gets predictions

        :param data: bunch
        :return: bunch with targetPredicted
        """
        data["targetPredicted"] = self.classifier.predict(data["data"])
        return data

    def print_classifier(self):
        """Prints the model/classifier
        """
        if self.classifier is None:
            return "Classifier not built yet."
        else:
            return str( self.classifier )

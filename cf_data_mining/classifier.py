#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

class Classifier(object):
    """
    Classifer class: wraps a Learner and a Model
    Needed if there is a need to use one of the widgets: "Build classifier", "Apply classifier", "Print model", "Cross-validation", etc.
    """

    def build_classifier(self, data):
        """Builds a classifier

        :param data: bunch
        """
        raise NotImplementedError

    def apply_classifier(self, data):
        """Applies a classifier on a dataset, and gets predictions

        :param data: bunch
        :return: data (bunch) with targetPredicted
        """
        raise NotImplementedError

    def print_classifier(self):
        """Prints the model/classifier
        """
        raise NotImplementedError

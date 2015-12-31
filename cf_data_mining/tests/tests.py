#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

import unittest
import cf_data_mining.classification as c
import cf_data_mining.regression as r
import cf_data_mining.evaluation as ev
import cf_data_mining.utilities as ut


class TestCFDataMining(unittest.TestCase):
    def test_classification_learners(self):
        """ Tests creating classification learners from the classification.py file
        :return: a list of learners
        """

        lrn = []
        num_exceptions = 0
        try:
            lrn.append(c.linear_SVC())
            lrn.append(c.logistic_regression())
            lrn.append(c.k_nearest_neighbors())
            lrn.append(c.J48())
            lrn.append(c.naive_bayes())
            lrn.append(c.SVC())

        except Exception, e:
            num_exceptions += 1
            print "Exception: " + str(e)

        self.assertIs(num_exceptions, 0)
        return lrn

    def test_regression_learners(self):
        """ Tests creating several regression learners from the regression.py file
        :return: a list of learners
        """

        lrn = []
        num_exceptions = 0
        try:
            lrn.append(r.svr())
            lrn.append(r.elastic_net_regression())
            lrn.append(r.ridge_regression())
            lrn.append(r.ard_regression())
            lrn.append(r.regression_tree())
            # lrn.append( r.lasso_LARS() ) # complains about data scaling ?
            # lrn.append( r.sgd_regressor() )# complains about data scaling ?

        except Exception, e:
            num_exceptions += 1
            print "Exception: " + str(e)

        self.assertIs(num_exceptions, 0)
        return lrn

    # --------------------------------------------------------------------------------------------------------------------------

    def test_classification_models(self):
        """ Tests building classification models using provided learners"""

        num_exceptions = 0
        lrn_arr = self.test_classification_learners()
        for lrn in lrn_arr:
            try:
                classification_dataset = ut.load_UCI_dataset("iris")

                ev.build_classifier(lrn, classification_dataset)

            except Exception, e:
                num_exceptions += 1
                print "Exception: " + str(e)

        self.assertIs(num_exceptions, 0)

    def test_regression_models(self):
        """ Tests building regression models using provided learners"""

        num_exceptions = 0
        lrn_arr = self.test_regression_learners()
        for lrn in lrn_arr:
            try:
                regression_dataset = ut.load_UCI_dataset("boston")

                ev.build_classifier(lrn, regression_dataset)

            except Exception, e:
                num_exceptions += 1
                print "Exception: " + str(e)

        self.assertIs(num_exceptions, 0)

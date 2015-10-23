__author__ = 'darkoa'

import unittest

import cf_datamining.classification as c, cf_datamining.regression as r, cf_datamining.unsupervised as u
import cf_datamining.evaluation as ev, cf_datamining.utilities as ut

class Tests_CF_Datamining(unittest.TestCase):


    def testClassificationLearners(self):
        """ Tests creating classification learners from the classification.py file
        :return: a list of learners
        """

        lrn = []
        try:
            lrn.append( c.linear_SVC() )
            lrn.append( c.logistic_regression() )
            lrn.append( c.k_nearest_neighbors() )
            lrn.append( c.k_nearest_neighbors() )
            lrn.append( c.J48() )
            lrn.append( c.naive_bayes() )
            lrn.append( c.SVC() )

        except Exception, e:
            print "Exception: " + e

        self.assertIs(len(lrn), 7)
        return lrn


    def testRegressionLearners(self):
        """ Tests creating several regression learners from the regression.py file
        :return: a list of learners
        """

        lrn = []
        try:
            lrn.append( r.svr() )
            lrn.append( r.elastic_net_regression() )
            lrn.append( r.ridge_regression() )
            lrn.append( r.ard_regression() )
            lrn.append( r.regression_tree() )
            # lrn.append( r.lasso_LARS() ) # complains about data scaling ?
            # lrn.append( r.sgd_regressor() )# complains about data scaling ?

        except Exception, e:
            print "Exception: " + e

        self.assertIs(len(lrn), 5)
        return lrn

# --------------------------------------------------------------------------------------------------------------------------

    def testClassificationModels(self):
        """ Tests building classification models using provided learners
        :return: True if all tests pass
        """

        lrn_arr =  self.testClassificationLearners()
        for lrn in lrn_arr:
            try:
                # lrn = c.linear_SVC(cIn=1.0, lossIn="l2", penaltyIn="l2", multiClassIn="ovr")

                # regressionDataset       = ut.load_UCI_dataset("boston")
                classificationDataset   = ut.load_UCI_dataset("iris")

                clf = ev.buildClassifier(lrn, classificationDataset)

            except Exception, e:
                clf = None
                print "Exception: " + e

            self.assertIsNotNone(clf)


    def testRegressionModels(self):
        """ Tests building regression models using provided learners
        :return: True if all tests pass
        """

        lrn_arr =  self.testRegressionLearners()
        for lrn in lrn_arr:
            try:
                # lrn = c.linear_SVC(cIn=1.0, lossIn="l2", penaltyIn="l2", multiClassIn="ovr")

                regressionDataset       = ut.load_UCI_dataset("boston")
                # classificationDataset   = ut.load_UCI_dataset("iris")

                clf = ev.buildClassifier(lrn, regressionDataset)

            except Exception, e:
                clf = None
                print "Exception: " + e

            self.assertIsNotNone(clf)


    def test_select_data_post(self):
        # [u'adviser', u'amdahl', u'apollo', u'basf', u'bti', u'burroughs', u'c.r.d', u'cdc', u'cambex', u'dec', u'dg', u'formation', u'four-phase', u'gould', u'hp', u'harris', u'honeywell', u'ibm', u'ipl', u'magnuson', u'microdata', u'nas', u'ncr', u'nixdorf', u'perkin-elmer', u'prime', u'siemens', u'sperry', u'sratus', u'wang']
        # [u'adviser', u'amdahl', u'apollo']
        i=1
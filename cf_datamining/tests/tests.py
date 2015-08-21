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
            lrn.append( c.linearSVC() )
            lrn.append( c.logisticRegression() )
            lrn.append( c.kNearestNeighbors() )
            lrn.append( c.kNearestNeighbors() )
            lrn.append( c.J48() )
            lrn.append( c.naiveBayes() )
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
            lrn.append( r.elasticNet() )
            lrn.append( r.ridge() )
            lrn.append( r.ardRegression() )
            lrn.append( r.decisionTreeRegressor() )
            # lrn.append( r.lassoLARS() ) # complains about data scaling ?
            # lrn.append( r.sgdRegressor() )# complains about data scaling ?

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
                # lrn = c.linearSVC(cIn=1.0, lossIn="l2", penaltyIn="l2", multiClassIn="ovr")

                # regressionDataset       = ut.loadUCIDataset("boston")
                classificationDataset   = ut.loadUCIDataset("iris")

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
                # lrn = c.linearSVC(cIn=1.0, lossIn="l2", penaltyIn="l2", multiClassIn="ovr")

                regressionDataset       = ut.loadUCIDataset("boston")
                # classificationDataset   = ut.loadUCIDataset("iris")

                clf = ev.buildClassifier(lrn, regressionDataset)

            except Exception, e:
                clf = None
                print "Exception: " + e

            self.assertIsNotNone(clf)

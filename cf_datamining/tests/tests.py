__author__ = 'darkoa'

import unittest

class TestLinearSVC(unittest.TestCase):

    def test_linearSVC(self):
        """ Tests a single classification method: linearSVC
        :return: True if all tests pass
        """
        import cf_datamining.classification as c
        clf = None
        try:
            clf = c.linearSVC(cIn=1.0, lossIn="l2", penaltyIn="l2", multiClassIn="ovr")
        except Exception, e:
            print "Exception: " + e

        self.assertIsNotNone(clf)

    def test_classification(self):
        """ Tests several classification methods from the classification.py file
        :return: True if all tests pass
        """

        import cf_datamining.classification as c
        clf = []
        try:
            clf.append( c.linearSVC() )
            clf.append( c.logisticRegression() )
            clf.append( c.kNearestNeighbors() )
            clf.append( c.kNearestNeighbors() )
            clf.append( c.J48() )

        except Exception, e:
            print "Exception: " + e

        self.assertIs(len(clf), 5)


    def test_regression(self):
        """ Tests several regression methods from the regression.py file
        :return: True if all tests pass
        """

        import cf_datamining.regression as r
        clf = []
        try:
            clf.append( r.svr() )
            clf.append( r.elasticNet() )
            clf.append( r.ridge() )
            clf.append( r.ardRegression() )
            clf.append( r.decisionTreeRegressor() )
            clf.append( r.lassoLARS() )
            clf.append( r.sgdRegressor() )

        except Exception, e:
            print "Exception: " + e

        self.assertIs(len(clf), 7)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

# ==========================================================================
# A dataset in ClowdFlows is a Bunch object (a dot-accessible dictionary),
#   as used in scikit-learn.
#
# dataset = Bunch(data=data,
#                    target=target,
#                    feature_names=[],
#                    DESCR="",
#                    target_names="")
# ==========================================================================

# these two functions does not work at all! (who wrote this stuff anyway?)
# remove or find a better way to test, e.g., number of different values of the feature

def is_target_nominal(bunch):
    return False
    # if 'target_names' in bunch and len(bunch.target_names)>0:
    #     #nominal target
    #     return True
    # else:
    #     return False

def is_feature_nominal(bunch, j):
    return False
    # if "feature_value_names" in bunch and len(bunch.feature_value_names[j])>0:
    #     #nominal feature
    #     return True
    # else:
    #     return False

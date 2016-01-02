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
def is_target_nominal(bunch):
    if bunch.has_key('target_names') and len(bunch.target_names)>0:
        #nominal target
        return True
    else:
        return False


def is_feature_nominal(bunch, j):
    if bunch.has_key("feature_value_names") and len(bunch.feature_value_names[j])>0:
        #nominal feature
        return True
    else:
        return False

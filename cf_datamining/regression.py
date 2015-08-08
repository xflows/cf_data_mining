__author__ = 'darkoa'

def decisionTreeRegressor(featureIn, depthIn):
    """Decision tree for regression problems

    :param featureIn: The number of features to consider when looking for the best split: If int, then consider max_features features at each split; If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split
    :param depthIn: The maximum depth of the tree
    :return: a DecisionTreeRegressor object
    """

    from sklearn import tree

    #parse input and determin its type
    try:
        featureValue= float(featureIn) if '.' in featureIn else int(featureIn) #return int or float
    except ValueError:
        featureValue= featureIn #return string
    
    clf = tree.DecisionTreeRegressor(max_features=featureValue, max_depth=int(depthIn))

    return clf


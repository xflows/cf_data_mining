__author__ = 'daleksovski'


def load_UCI_dataset(dsIn):
    '''Loads a UCI dataset

    :param dsIn: the dataset name
    :return: A SciKit dataset
    '''

    from sklearn import datasets
    allDSets = {"iris":datasets.load_iris(), "boston":datasets.load_boston(), "diabetes":datasets.load_diabetes(), " linnerud":datasets.load_linnerud()}
    dataset = allDSets[dsIn]
    return dataset

__author__ = 'darkoa'

def k_means(instances, k):
    """k-Means clustering

    :param instances: data instances
    :param k: num.clusters
    :return: ( clusterCenters , clusteredData )
    """
    data = instances
    X = data['data']      # dsNum['dtsOut']['data']
    k = int( k )

    from sklearn.cluster import KMeans
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
    # t0 = time.time()
    k_means.fit(X)
    # t_batch = time.time() - t0
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    # k_means_labels_unique = np.unique(k_means_labels)

    data["cluster_id"] = k_means_labels

    return (k_means_cluster_centers, data)  #  clusterCenters , clusteredData


def aglomerative_clustering(instances, k):
    """Hierarchical Agglomerative Clustering, using the Ward linkage and euclidean metric.

    :param instances: data instances
    :param k: num.clusters, default value 3.
    :return: clusteredData
    """

    data = instances
    X = data['data']      # dsNum['dtsOut']['data']
    n_clusters = int( k )

    from sklearn.cluster import AgglomerativeClustering

    metric = "euclidean"    #["cosine", "euclidean", "cityblock"]
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", affinity=metric)
    model.fit(X)
    agl_clust_labels = model.labels_

    data["cluster_id"] = agl_clust_labels

    return data
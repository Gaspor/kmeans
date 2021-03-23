import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def k_means(data, clusterNumber):
    dotSizeCluster = 100

    kmeans = KMeans(clusterNumber, init='k-means++', n_init=10, max_iter=300)
    pred_y = kmeans.fit_predict(data)
    plt.scatter(data[:, 1], data[:, 0], c=pred_y)
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.grid()
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], dotSizeCluster, c='red')
    plt.show()


dataset = np.array([[25,  4],
                    [22,  3],
                    [ 5,  9],
                    [ 3, 11],
                    [19, 43],
                    [15, 47],
                    [12,  8],
                    [ 8, 21],
                    [ 6,  9],
                    [ 3, 20],
                    [22, 47],
                    [ 3, 38],
                    [21, 47],
                    [ 6, 31],
                    [ 2, 14],
                    [21, 43],
                    [ 1,  8],
                    [48, 47],
                    [28,  5],
                    [32,  2],
                    [14, 36]])

clusters = 3

k_means(dataset, clusters)

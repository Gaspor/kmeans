import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

startPlot = 0
endPlot = 50
clusters = 3


def k_means(start, end, data, clusterNumber):
    dotSizeCluster = 25

    kmeans = KMeans(clusterNumber, init='k-means++', n_init=10, max_iter=300)
    pred_y = kmeans.fit_predict(data)

    plt.scatter(data[:, 1], data[:, 0], 10, c=pred_y)
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], dotSizeCluster, c='red')

    plt.xlim(start, end)
    plt.ylim(start, end)

    plt.grid()
    plt.show()


def getDataSet(beginInterval, endInterval, elements):
    return np.random.randint(beginInterval, endInterval, (elements, 2))


dataset = getDataSet(startPlot, endPlot, 60)
k_means(startPlot, endPlot, dataset, clusters)

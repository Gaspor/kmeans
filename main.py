import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

startPlot = 0
endPlot = 200
clusters = 3


def k_means(start, end, data, clusterNumber):
    dotSizeCluster = 25

    kmeans = KMeans(clusterNumber, init='k-means++', n_init=10, max_iter=300)
    pred_y = kmeans.fit_predict(data)

    plt.scatter(data[:, 0], data[:, 1], 10, c=pred_y)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], dotSizeCluster, c='red')

    plt.xlim(start, end)
    plt.ylim(start, end)

    plt.grid()
    plt.show()


def getDataSet(beginInterval, endInterval, numberElements):
    setDataset = np.random.randint(beginInterval, endInterval, (numberElements, 2))
    return setDataset


dataset = getDataSet(startPlot, endPlot, numberElements = 20)
k_means(startPlot, endPlot, dataset, clusters)

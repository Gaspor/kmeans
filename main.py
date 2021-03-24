import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def main():
    startPlot = 0
    endPlot = 200
    clusters = 3

    dataset = getDataSet(startPlot, endPlot, numberElements=20)
    k_means(startPlot, endPlot, dataset, clusters)


def k_means(start, end, data, clusterNumber):
    dotSize = 10
    dotSizeCluster = 25

    kmeans = KMeans(clusterNumber, init='k-means++', n_init=10, max_iter=300)
    pred_y = kmeans.fit_predict(data)

    plt.scatter(data[:, 0], data[:, 1], dotSize, c=pred_y)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], dotSizeCluster, c='red')

    plt.xlim(start, end)
    plt.ylim(start, end)

    plt.grid()
    plt.show()


def getDataSet(beginInterval, endInterval, numberElements):
    setDataset = np.random.randint(beginInterval, endInterval, (numberElements, 2))
    return setDataset


if __name__ == '__main__':
    main()
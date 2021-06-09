import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def main():
    dataset = pd.read_csv('model.txt', sep="\t", usecols=[0, 1], names=['X', 'Y'])
    df = pd.DataFrame(dataset, columns=['X', 'Y'])

    startPlot = (df['X'].min()) - 2 if df['X'].min() < df['Y'].min() else (df['Y'].min()) - 2
    endPlot = (df['X'].max()) + 2 if df['X'].max() > df['Y'].max() else (df['Y'].max()) + 2

    clusters = getClusters(df)
    k_means(startPlot, endPlot, dataset, clusters)
    dbscan(df)


def k_means(start, end, data, clusterNumber):
    dotSize = 10
    dotSizeCluster = 25

    kmeans = KMeans(clusterNumber, init = 'k-means++', n_init = 10, max_iter = 1500, tol = 0.0004)
    pred_y = kmeans.fit_predict(data)

    data.plot.scatter('X', 'Y', dotSize, c=pred_y, cmap='turbo')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], dotSizeCluster, c = 'red')

    plt.xlim(start, end)
    plt.ylim(start, end)

    plt.grid()
    plt.show()


def getClusters(dataset):
    distortions = []
    kNumber = 1
    aux = 0
    K = range(1, 30)
    for k in K:
        kmeanModel = KMeans(n_clusters = k, init = 'random', max_iter = 500)
        kmeanModel.fit(dataset)
        distortions.append(kmeanModel.inertia_)

        if k > 1:
            aux = ((distortions[0] - distortions[1]) / 100) * 8

        if distortions[k - 2] - distortions[k - 1] > aux:
            kNumber = k

    return kNumber


def dbscan(data):
    X = StandardScaler().fit_transform(data)
    db = DBSCAN(eps = 0.289, min_samples=5).fit(X)
    labels = db.labels_

    color_list = np.array(['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple'])
    plt.scatter(X[:, 0], X[:, 1], c=color_list[labels])

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

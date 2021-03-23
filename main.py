import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

dataset = np.array([
[25,  4],
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
[ 2, 31],
[ 2, 14],
[21, 43],
[ 1,  8],
[ 1, 47],
[ 8, 2]])

kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init = 10,max_iter = 300)
pred_y = kmeans.fit_predict(dataset)
plt.scatter(dataset[:,1], dataset[:,0], c = pred_y)
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.grid()
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s = 70, c = 'red')
plt.show()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [1,2],
    [2,1],
    [3,2],
    [10,10],
    [11,9],
    [12,10]
])

model = KMeans(
    n_clusters=2,
    random_state=42
)

model.fit(X)

plt.scatter(
    X[:,0],
    X[:,1],
    c=model.labels_
)

plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    marker='X',
    s=200
)

plt.show()

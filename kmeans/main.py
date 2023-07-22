import numpy as np
from sklearn.datasets import make_blobs
from kmeans import KMeans 

X, y = make_blobs(centers=5, n_samples=500, n_features=2, shuffle=True, random_state=42)
print(X.shape)

clusters = len(np.unique(y))

k = KMeans(clusters, 150, True)
print(k.predict(X))

k.plot()


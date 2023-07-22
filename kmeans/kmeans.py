import numpy as np
import matplotlib.pyplot as plt

np.random.seed(24)

def distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = 5
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for i in range(self.K)]
        self.centers = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # init clusters
        center_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centers = [self.X[x] for x in center_idxs]
    
        for _ in range(self.max_iters):
            # split to clusters
            self.clusters = self._create_clusters(self.centers)
            if self.plot_steps:
                self.plot()
            old_centers = self.centers
            # calculate new centers
            self.centers = self._get_centers(self.clusters)
            if self.plot_steps:
                self.plot()
            # check whether stop
            if self._is_finished(old_centers, self.centers):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for idx, cluster in enumerate(self.clusters):
            for v in cluster:
                labels[v] = idx
        return labels

    def _create_clusters(self, centers):
        clusters = [[] for i in range(self.K)]
        for i, sample in enumerate(self.X):
            distances = [distance(sample, center) for center in centers]
            clusters[np.argmin(distances)].append(i)
        return clusters

    def _get_centers(self, clusters):
        centers = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            centers[idx] = np.mean(self.X[cluster], axis=0)
        return centers

    def _is_finished(self, old, new):
        distances = [distance(old[i], new[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centers:
            ax.scatter(*point, marker='x', color='black', linewidth=2)
        plt.show()

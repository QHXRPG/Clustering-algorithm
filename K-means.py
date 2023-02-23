from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

data = load_iris()
x = data.data
k_means = KMeans(n_clusters=3)
k_means.fit(x)
labels = k_means.labels_
print(labels)

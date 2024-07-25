# 导入所需的库
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from ucimlrepo import fetch_ucirepo

# 生成模拟数据
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 可视化生成的数据
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated Data")
plt.show()

# 应用 K-means 聚类
kmeans = KMeans(n_clusters=4)  # 指定聚类中心数量
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
#scatter表示是plt中的散点图画法

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)  # 标记聚类中心
plt.title("K-means Clustering Results")
plt.show()

x_i,y_i=[],[]
for i in range(1,11):
    n_clusters = int(i)#首先测试的超参数\
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    inertia = cluster.inertia_
    x_i.append(i)
    y_i.append(inertia)
plt.plot(y_i)
plt.show()

from sklearn.metrics import silhouette_score
x_i,y_i=[],[]
for n_clusters in range(2,11):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    x_i.append(n_clusters)
    y_i.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
plt.plot(x_i, y_i)
plt.show()



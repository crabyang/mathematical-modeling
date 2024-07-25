from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# data = pd.read_csv(url, names=columns)
# X = data.iloc[:, :-1]  # 使用花的测量数据

data = pd.read_csv('C:/Users/26962/PycharmProjects/MachineLearning/Wholesale customers data.csv')
# X = data.drop(data.columns[1], axis=1)
X = data.iloc[:,2:]
y = data.columns[1]

# 应用 K-means 聚类
kmeans = KMeans(n_clusters=3)  # 指定聚类中心数量
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 使用 PCA 降维到二维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 可视化聚类结果
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_kmeans, s=50, cmap='viridis')
#scatter表示是plt中的散点图画法

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)  # 标记聚类中心
plt.title("K-means Clustering Results")
plt.show()



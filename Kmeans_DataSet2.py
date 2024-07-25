import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 加载数据
data_path = '/Wholesale customers data.csv'
data = pd.read_csv(data_path)

# 假设我们关注所有数值特征进行聚类
X = data.select_dtypes(include=[np.number])

# 数据预处理 - 规范化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用 K-means 聚类
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# 使用 PCA 降维以便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化聚类结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k', alpha=0.7, s=50)
plt.title('K-means Clustering of Wholesale Customers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter)
plt.show()

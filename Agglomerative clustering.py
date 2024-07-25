import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# 生成一些随机数据
X, labels_true = make_blobs(n_samples=150, centers=3, cluster_std=0.60, random_state=0)

# 使用 Ward 的方法进行层次聚类
linked = linkage(X, 'ward')

# 绘制树状图
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=labels_true,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

from scipy.cluster.hierarchy import fcluster

# 选择一个距离阈值来切割树，这里我们以10为阈值
clusters = fcluster(linked, 10, criterion='distance')
print("Cluster memberships:\n", clusters)


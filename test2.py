import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 选择固定值的特征
fixed_features = np.mean(X[:, :2], axis=0)  # 计算前两个特征的平均值

# 训练分类器
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# 生成网格点，专注于第三个和第四个特征
feature_x_index = 2  # 第三个特征
feature_y_index = 3  # 第四个特征
x_min, x_max = X[:, feature_x_index].min() - 1, X[:, feature_x_index].max() + 1
y_min, y_max = X[:, feature_y_index].min() - 1, X[:, feature_y_index].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# 生成新的测试数据集
# 每个点都固定了前两个特征的值
test_data = np.c_[fixed_features[0]*np.ones(xx.ravel().shape),
                  fixed_features[1]*np.ones(xx.ravel().shape),
                  xx.ravel(),
                  yy.ravel()]

# 预测新的测试数据集
Z = clf.predict(test_data)
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, feature_x_index], X[:, feature_y_index], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.title('Decision Boundary with Fixed Features')
plt.xlabel(data.feature_names[feature_x_index])
plt.ylabel(data.feature_names[feature_y_index])
plt.show()

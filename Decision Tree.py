import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

maxDepth = 4    #超参数的设置   这里是最大树深度，放置出现过拟合的状况
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建决策树分类器实例
dtree = DecisionTreeClassifier(random_state=42,max_depth = maxDepth)

# 训练模型
dtree.fit(X_train, y_train)
# 可视化决策树
plt.figure(figsize=(20,10))
tree.plot_tree(dtree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
# 使用测试集进行预测
predictions = dtree.predict(X_test)
# 评估模型性能
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, predictions))

# 调用函数查看概率估计
#dtree.predict_proba([[5,1.5]])
# 调用函数查看预测结果
#dtree.predict([[5,1.5,3,5]])


# 使用 PCA 降维到二维
# pca = PCA(n_components=2)
# X_train = pca.fit_transform(X_train)

def plot_decision_boundary(classifier, X, y, feature_indices, fixed_value='mean', resolution=1000):
    """
    Plots detailed decision boundaries for a classifier based on specified feature indices.

    Args:
    classifier: A trained classifier.
    X: Full feature dataset (numpy array).
    y: Labels (numpy array).
    feature_indices: Indices of the features to visualize.
    fixed_value: Strategy for non-visualized features ('mean', 'median', or specific value as np.array).
    resolution: The number of divisions per axis in the grid.
    """
    if fixed_value == 'mean':
        fixed_values = np.mean(X, axis=0)
    elif fixed_value == 'median':
        fixed_values = np.median(X, axis=0)
    else:
        fixed_values = fixed_value

    # Setting ranges for the feature grid
    x_min, x_max = X[:, feature_indices[0]].min() - 1, X[:, feature_indices[0]].max() + 1
    y_min, y_max = X[:, feature_indices[1]].min() - 1, X[:, feature_indices[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Create grid data with fixed values for other features
    grid_data = np.tile(fixed_values, (xx.size, 1))
    grid_data[:, feature_indices[0]] = xx.ravel()
    grid_data[:, feature_indices[1]] = yy.ravel()

    # Predictions for the grid data
    Z = classifier.predict(grid_data)
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)  # Draw boundary lines
    plt.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature ' + str(feature_indices[0]))
    plt.ylabel('Feature ' + str(feature_indices[1]))
    plt.title('Decision Boundary Visualization')
    plt.show()

plot_decision_boundary(dtree, X, y, [2, 3])

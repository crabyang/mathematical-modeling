import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

maxDepth = 2    #超参数的设置   这里是最大树深度，放置出现过拟合的状况
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


# # 使用 PCA 降维到二维
# pca = PCA(n_components=2)
# X_train = pca.fit_transform(X_train)


#-----------------决策边界函数处理部分-------------------------------------------
from matplotlib.colors import ListedColormap
# 定义绘制决策边界的函数

def plot_decision_boundary(clf):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # 选择要动态变化的两个特征索引，这里选择花瓣长度和宽度（索引2和3）
    feature_x_index = 2
    feature_y_index = 3

    # 选择固定值的特征，这里固定花萼长度和宽度（索引0和1）
    # 使用平均值作为固定值
    fixed_features = np.mean(X[:, :2], axis=0)

    # 构建完整的测试集
    # 这里每一个网格点都复制相同的固定特征值
    test_data = np.c_[np.full((xx.size, 2), fixed_features), xx.ravel(), yy.ravel()]
    # 预测网格点的类别
    Z = clf.predict(test_data)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()




def plot_decision_boundary2(clf, X, y, axes=[0, 7, 0, 3], iris=True, legend=False, plot_training=True):
    # 构建坐标棋盘
    # 等距选 100 个居于 axes[0],axes[1] 之间的点
    x1s = np.linspace(axes[0], axes[1], 100)
    # x1s.shape = (100,)

    # 等距选 100 个居于 axes[2],axes[3] 之间的点
    x2s = np.linspace(axes[2], axes[3], 100)
    # x2s.shape = (100,)

    # 构建棋盘数据
    x1, x2 = np.meshgrid(x1s, x2s)
    # x1.shape = x2.shape = (100,100)

    # 选择要动态变化的两个特征索引，这里选择花瓣长度和宽度（索引2和3）
    feature_x_index = 2
    feature_y_index = 3

    # 选择固定值的特征，这里固定花萼长度和宽度（索引0和1）
    # 使用平均值作为固定值
    fixed_features = np.mean(X[:, :2], axis=0)

    # 构建完整的测试集
    # 这里每一个网格点都复制相同的固定特征值
    test_data = np.c_[np.full((x1.size, 2), fixed_features), x1.ravel(), x2.ravel()]

    # 将构建好的两个棋盘数据分别作为一个坐标轴上的数据（从而构成新的测试数据）
    # x1.ravel() 将拉平数据（得到的是个列向量（矩阵）），此时 x1.shape = (10000,)
    # 将 x1 和 x2 拉平后分别作为两条坐标轴
    # 这里用到 numpy.c_() 函数，以将两个矩阵合并为一个矩阵
    #X_new = np.c_[x1.ravel(), x2.ravel()]
    # 此时 X_new.shape = (10000,2)

    # 对构建好的新测试数据进行预测
    y_pred = clf.predict(test_data).reshape(x1.shape)

    # 选用背景颜色
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

    # 执行填充
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)


    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


plt.figure(figsize=(8, 4))
plot_decision_boundary(dtree)
plt.show()

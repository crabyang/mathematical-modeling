#本项目的代码主干来自于https://blog.csdn.net/weixin_38753213/article/details/103676065?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-103676065-blog-130089515.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-103676065-blog-130089515.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=10

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 加载葡萄酒的数据集
wine = datasets.load_wine()

# 为了方便可视化，只选取 2 个特征
X = wine.data[:, [0, 6]]
y = wine.target

# 绘制散点图
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn import tree

# 拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 调用决策树分类算法
dtc = tree.DecisionTreeClassifier(max_depth=2)
dtc.fit(X_train, y_train)

# 算法评分
print('训练得分：', dtc.score(X_train, y_train))
print('测试得分：', dtc.score(X_test, y_test))

from matplotlib.colors import ListedColormap


# 定义绘制决策边界的函数
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)


# 绘制决策边界
plot_decision_boundary(dtc, axis=[11, 15, 0, 6])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.show()
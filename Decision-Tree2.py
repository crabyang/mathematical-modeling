# 导入鸢尾花数据集 和 决策树的相关包
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 加载鸢尾花数据集
iris = load_iris()

# 选用鸢尾花数据集的特征
# 尾花数据集的 4 个特征分别为：sepal length:、sepal width、petal length:、petal width
# 下面选用 petal length、petal width 作为实验用的特征
X= iris.data[:,2:]

# 取出标签
y = iris.target

# 设置决策树的最大深度为 2（也可以限制其他属性）
tree_clf = DecisionTreeClassifier(max_depth = 2)

# 训练分类器
tree_clf.fit(X, y)

# 导入对决策树进行可视化展示的相关包
from sklearn.tree import export_graphviz

export_graphviz(
    # 传入构建好的决策树模型
    tree_clf,

    # 设置输出文件（需要设置为 .dot 文件，之后再转换为 jpg 或 png 文件）
    out_file="iris_tree.dot",

    # 设置特征的名称
    feature_names=iris.feature_names[2:],

    # 设置不同分类的名称（标签）
    class_names=iris.target_names,

    rounded=True,
    filled=True
)

# 该代码执行完毕后，此 .ipython 文件存放路径下将生成一个 .dot 文件（名字由 out_file 设置，此处我设置的文件名为 iris_tree.dot）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def plot_decision_boundary(classifier, X, y, feature_indices, fixed_value='mean'):
    """
    Plots decision boundaries for a classifier based on specified feature indices.

    Args:
    classifier: A trained classifier (e.g., RandomForestClassifier).
    X: Full feature dataset (numpy array).
    y: Labels (numpy array).
    feature_indices: Indices of the features to visualize.
    fixed_value: Strategy for non-visualized features ('mean', 'median', or specific value as np.array).
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
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Create grid data with fixed values for other features
    grid_data = np.tile(fixed_values, (xx.size, 1))
    grid_data[:, feature_indices[0]] = xx.ravel()
    grid_data[:, feature_indices[1]] = yy.ravel()

    # Predictions for the grid data
    Z = classifier.predict(grid_data)
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature ' + str(feature_indices[0]))
    plt.ylabel('Feature ' + str(feature_indices[1]))
    plt.colorbar(scatter)
    plt.title('Decision Boundary Visualization')
    plt.show()


# Example usage
data = load_iris()
X = data.data
y = data.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Plot decision boundary for the first and second features
plot_decision_boundary(model, X, y, [0, 1])

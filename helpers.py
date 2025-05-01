import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model_method, X, y):
    """
    Plots the decision boundary for a model using a 2D dataset.

    Parameters:
    - model_method: function that takes (N, 2) input and returns (N,) predictions.
    - X: Input data of shape (N, 2).
    - y: Labels of shape (N,).
    """
    # Set min and max values for each feature and add some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # Step size for grid

    # Create a mesh grid covering the input space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict values over the entire grid
    Z = model_method(np.c_[xx.ravel(), yy.ravel()])  # Pass reshaped (M, 2) input
    Z = Z.reshape(xx.shape)  # Reshape to match the grid shape

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # Plot the training examples
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Decision Boundary")
    plt.show()

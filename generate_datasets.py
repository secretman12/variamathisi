import numpy as np


def generate_binary_problem(centers: np.ndarray, N: int = 100):
    """
    Generate a set of 2D points belonging in two classes

    N: int. Number of samples per class
    p: int. Number of dimensions
    centers: numpy.ndarray. A matrix whose columns correspond to the center
             of each class. Unit covariance matrix is assumed for all classes
    """

    rng = np.random.default_rng()
    # Class 0
    X0 = rng.multivariate_normal(centers[:, 0], np.eye(2), N)
    y0 = np.zeros(N)
    # Class 1
    X1 = rng.multivariate_normal(centers[:, 1], np.eye(2), N)
    y1 = np.ones(N)
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    return X, y


def generate_flower_problem():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y.ravel()



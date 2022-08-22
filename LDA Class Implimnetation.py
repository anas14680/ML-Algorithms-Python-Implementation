"""Bio_stats Assignment 5 - Mohammad Anas."""

# import required packages
import numpy as np
from numpy.random import multivariate_normal
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def generate_data(
    c1_mean: Union[int, float],
    c1_cov: np.ndarray,
    c2_mean: Union[int, float],
    c2_cov: np.ndarray,
    n1: int,
    n2: int,
):
    """This funnction generates a dataset.

    The data generated is based on the distribution
    parameters, two class labels generated with two
    features.
    """
    X1 = multivariate_normal(c1_mean, c1_cov, n1)
    y1 = np.ones(n1)
    X2 = multivariate_normal(c2_mean, c2_cov, n2)
    y2 = np.zeros(n2)
    X = np.vstack((X1, X2))
    y = np.append(y1, y2)
    return X, y


class LDA:
    """LDA Class impliments fishers LDA model on a data."""

    def __init__(self, n_disc: int) -> None:
        """Instantiate LDA class.

        The user must state the number of linear discriminants
        to project data on."""
        self.n_disc = n_disc
        # w here is our parameter vector
        # that needs to be optimized.
        self.w: Union[None, np.ndarray] = None
        # This attributes contains the projected data
        self.projections: Union[None, np.ndarray] = None
        # This attribute the projected threshold to
        # seperate classes
        self.thresh: Union[None, int, float, np.ndarray] = None

    def within_group_var(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """This method computes with variance of classes."""
        # seperate each class data
        sample0 = X_train[y_train == 0]
        sample1 = X_train[y_train == 1]
        # calculate number of datapoints in each class
        n_sample0 = sample0.shape[0]
        n_sample1 = sample1.shape[0]
        # find covariances features of each class
        cov_1 = np.cov(sample1.T)
        cov_0 = np.cov(sample0.T)
        # find pooled within variance
        pooled_within_var = ((n_sample0 * cov_0) + (n_sample1 * cov_1)) / (
            n_sample0 + n_sample1 - 2
        )
        return pooled_within_var

    def between_group_var(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Find between group variance."""
        # get within variance
        within_var = self.within_group_var(X_train, y_train)
        # find complete covvariance of the data
        total_cov = np.cov(X_train.T)
        # calculate covariance of between groups
        between_var = total_cov - within_var
        return between_var

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the classifier to the data."""
        # find within and between group variances
        within_cov = self.within_group_var(X_train, y_train)
        between_cov = self.between_group_var(X_train, y_train)
        # find the vector that we get from differentation
        # cost function with repect to w
        opt_vec = np.linalg.inv(within_cov) @ between_cov
        # calculate eigen vectors and eigen values
        eig_val, eig_vec = np.linalg.eig(opt_vec)
        # get the required number of eigenvectors based on
        # the eigen values. Numpy return eigen values sorted in
        # decreasing order. These eigen vectors define projections
        self.w = eig_vec[:, : self.n_disc]
        self.projections = X_train @ self.w
        # once data is projected we calculate the threshhold to
        # seperate the data. Note that this threshhold is generally
        # a hyperparamter that needs to be optimized.
        # However, we have
        # set an ideal one, which mostly works fine
        # In the predict function, the user has the liberty to set
        # this threshhold, but if not set the defined threshhold is used.
        cl1_proj_mean = np.mean(self.projections[y_train == 1])
        cl0_proj_mean = np.mean(self.projections[y_train == 0])
        self.thresh = (cl1_proj_mean + cl0_proj_mean) / 2

    def predict(
        self, X_test: np.ndarray, thresh: Union[int, float, np.ndarray] = None
    ) -> np.ndarray:
        """Make predictions on test data."""
        # calculate projections
        test_projections = X_test @ self.w

        # make classifications based on the threshhold
        if thresh is None:
            predicted = np.where(test_projections > self.thresh, 1, 0)
            return predicted
        else:
            predicted = np.where(test_projections > thresh, 1, 0)
            return predicted


def plot_scatter_desicion(
    X: np.ndarray, y: np.ndarray, classifer: LDA, title: str, fontsize: int
):
    """This function creates scatter plot and desicion boundary.

    It take the data and the Trained Classifier as arguements."""
    # this h is the step size we late use in meshgrid
    h = 0.02
    cmap_light = ListedColormap(["cyan", "yellow"])

    # seperate the class label data
    sample1 = X[y == 1]
    sample0 = X[y == 0]
    # make scatter plots of each class label
    plt.scatter(sample1[:, 0], sample1[:, 1], label=1, color="#FFCD01", marker="^")
    plt.scatter(sample0[:, 0], sample0[:, 1], label=0, color="#1357a6")
    # prepare min and max axis for desicion boundary
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # make all possible combinations
    x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    boundary_pred = np.c_[x1x1.ravel(), x2x2.ravel()]
    # make prediction on all possible point
    boundary = classifer.predict(boundary_pred).reshape(x1x1.shape)
    # fill colors accordingly
    plt.contourf(x1x1, x2x2, boundary, cmap=cmap_light, alpha=0.1)
    plt.title(title, fontsize=fontsize + 2)
    plt.xlabel("X1 feature", fontsize=fontsize)
    plt.ylabel("X2 feature", fontsize=fontsize)
    plt.legend()


if __name__ == "__main__":
    np.random.seed(255)
    n1 = 320
    n2 = 280
    c1_mean = np.array([2, 3])
    c2_mean = np.array([-7, 9])
    c1_cov = np.array([[9, -1], [-1, 8]])
    c2_cov = np.array([[2, 1], [1, 4]])

    X, y = generate_data(c1_mean, c1_cov, c2_mean, c2_cov, n1, n1)
    model = LDA(1)
    model.fit(X, y)
    plt.figure(figsize=(8, 8))
    plot_scatter_desicion(X, y, model, "Class Label Seperation", 15)

import numpy as np
import pandas as pd


### define Node


class Node:
    def __init__(self, X, y, feature, thresh):
        self.X = X
        self.y = y
        self.feature = feature
        self.thresh = thresh
        self.left = None
        self.right = None

        if len(np.unique(y)) == 1:
            self.leaf = True
        else:
            self.leaf = False

    def isLeafNode(self):
        return self.leaf

    def samples_in_node(self):
        return len(self.y)


## Desicion Tree Classifier


class desiciontree:
    def __init__(self, min_sample_split=2, max_depth=100, criterion="entropy"):
        self.root = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.criterion = criterion
        assert self.criterion in {"gini", "entropy"}

    def entropy(self, y):

        value, counts = np.unique(y, return_counts=True)
        probs = counts / y.shape[0]
        log_probs = np.log2(probs)
        entropy = sum(probs * log_probs) * -1
        return entropy

    def gini(self, y):

        value, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        gini = 1 - np.sum(probs ** 2)
        return gini

    def information_gain(self, feature, thresh, X, y):
        split1 = y[X[:, feature] > thresh]
        split2 = y[X[:, feature] <= thresh]

        if self.criterion == "gini":
            num = (len(split1) * self.gini(split1)) + (len(split2) * self.gini(split2))
            den = len(y)
            gini_feature = num / den
            return self.gini(y) - gini_feature

        if self.criterion == "entropy":
            num = (len(split1) * self.entropy(split1)) + (
                len(split2) * self.entropy(split2)
            )
            den = len(y)

            entropy_feature = num / den
            return self.entropy(y) - entropy_feature

    def best_split(self, X, y):
        best_feature = None
        best_thresh = None
        highest_gain = -99999

        for feature in range(X.shape[1]):
            unique_threshs = np.unique(X[:, feature])
            for thresh in unique_threshs:

                gain = self.information_gain(feature, thresh, X, y)
                if gain > highest_gain:
                    highest_gain = gain
                    best_feature = feature
                    best_thresh = thresh
                else:
                    pass

        return (best_feature, best_thresh)

    def grow_tree(self, X, y, depth=0):

        node_feature, node_thresh = self.best_split(X, y)
        temp_node = Node(X, y, node_feature, node_thresh)

        if temp_node.isLeafNode():
            return temp_node
        if temp_node.samples_in_node() <= self.min_sample_split:
            return temp_node
        if depth >= self.max_depth:
            return temp_node

        X_left = X[X[:, node_feature] <= node_thresh]
        y_left = y[X[:, node_feature] <= node_thresh]

        X_right = X[X[:, node_feature] > node_thresh]
        y_right = y[X[:, node_feature] > node_thresh]

        temp_node.left = self.grow_tree(X_left, y_left, depth + 1)
        temp_node.right = self.grow_tree(X_right, y_right, depth + 1)

        return temp_node

    def fit(self, X, y):
        self.root = self.grow_tree(X, y)

    def mode(self, y):
        values, counts = np.unique(y, return_counts=True)

        mode_value = 0
        max_count = 0
        for i in range(len(values)):
            if counts[i] > max_count:
                max_count = counts[i]
                mode_value = values[i]

        return mode_value

    def make_prediction(self, x, node):

        if (node.left is None) and (node.right is None):
            return self.mode(node.y)

        if x[node.feature] > node.thresh:
            return self.make_prediction(x, node.right)

        if x[node.feature] <= node.thresh:
            return self.make_prediction(x, node.left)

    def predict(self, X):

        predictions = []

        for i in range(X.shape[0]):
            predictions.append(self.make_prediction(X[i, :], self.root))
        return np.array(predictions)

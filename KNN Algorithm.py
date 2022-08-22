# Skeleton code for part (a) to write your own kNN classifier
import numpy as np


class Knn:
    # k-Nearest Neighbor class object for classification training and testing
    def __init__(self):

        self.x = None
        self.y = None
        self.test_x = None

    def fit(self, x, y):
        self.x = x
        self.y = y

    # Save the training data to properties of this class

    def predict(self, x, k):

        self.test_x = x
        y_unique = np.unique(self.y)

        y_hat = []
        # Variable to store the estimated class label for test data
        # Description of code below
        # Calculate the distance from each vector in test x to the training data
        # use loop to go over each point in test data
        # numpy facilitates substracting one array from the whole train data
        # therefore we only had to use one loop. We subtract test array from
        # train array and square (each element) of that array and then compute the square root
        # of the sum of that array.

        for i in range(self.test_x.shape[0]):
            a = (self.x - self.test_x[i]) ** 2
            distance_sq = np.sum(a, axis=1)
            distance = np.sqrt(distance_sq)

            #  using argsort we get the index of arrays with least distances
            sorted_ind = np.argsort(distance)[:k]
            k_yvals = self.y[sorted_ind]
            count_y1 = len(k_yvals[k_yvals == y_unique[0]])
            # we count the number of time each label occurs in our array
            count_y2 = len(k_yvals[k_yvals == y_unique[1]])

            # Then we count the labels and assign accordingly
            # based on count of labels we assign them and add them to y-hat
            if count_y1 > count_y2:
                y_hat.append(y_unique[0])
            else:
                y_hat.append(y_unique[1])

        # Return the estimated targets
        return y_hat

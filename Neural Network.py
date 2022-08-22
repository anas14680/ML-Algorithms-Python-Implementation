import numpy as np


class myNeuralNetwork(object):
    def __init__(self, n_in, n_layer1, n_layer2, n_out, learning_rate=0.01):
        """__init__
        Class constructor: Initialize the parameters of the network including
        the learning rate, layer sizes, and each of the parameters
        of the model (weights, placeholders for activations, inputs,
        deltas for gradients, and weight gradients). This method
        should also initialize the weights of your model randomly
            Input:
                n_in:          number of inputs
                n_layer1:      number of nodes in layer 1
                n_layer2:      number of nodes in layer 2
                n_out:         number of output nodes
                learning_rate: learning rate for gradient descent
            Output:
                none
        """
        self.lr = learning_rate

        np.random.seed(231)
        self.weights = [
            np.random.random((n_layer1, n_in)),
            np.random.random((n_layer2, n_layer1)),
            np.random.random((n_out, n_layer2)),
        ]
        self.deltas = [None, None, None]
        self.w_gradients = [None, None, None]
        self.activations = [None, None, None]
        self.act_sig = [None, None, None]
        self.biases = [np.zeros((5, 1)), np.zeros((5, 1)), np.zeros((1, 1))]
        self.bias_grad = [None, None, None]

    def sigmoid(self, X):
        """sigmoid
        Compute the sigmoid function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid function
        """
        numerator = 1
        denominator = 1 + np.exp(-X)
        return numerator / denominator

    def sigmoid_derivative(self, X):
        """sigmoid_derivative
        Compute the sigmoid derivative function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid derivative function
        """
        sig = self.sigmoid(X)
        sig2 = 1 - self.sigmoid(X)
        return sig * sig2

    def forward_propagation(self, x):
        """forward_propagation
        Takes a vector of your input data (one sample) and feeds
        it forward through the neural network, calculating activations and
        layer node values along the way.
            Input:
                x: a vector of data representing 1 sample [n_in x 1]
            Output:
                y_hat: a vector (or scaler of predictions) [n_out x 1]
                (typically n_out will be 1 for binary classification)
        """
        self.activations[0] = (self.weights[0] @ x) + self.biases[0]
        self.act_sig[0] = self.sigmoid(self.activations[0])

        self.activations[1] = (self.weights[1] @ self.act_sig[0]) + self.biases[1]
        self.act_sig[1] = self.sigmoid(self.activations[1])

        self.activations[2] = (self.weights[2] @ self.act_sig[1]) + self.biases[2]
        self.act_sig[2] = self.sigmoid(self.activations[2])

        return self.act_sig[2]

    def predict_proba(self, X):
        """predict_proba
        Compute the output of the neural network for each sample in X, with the last layer's
        sigmoid activation providing an estimate of the target output between 0 and 1
            Input:
                X: A matrix of N samples of data [N x n_in]
            Output:
                y_hat: A vector of class predictions between 0 and 1 [N x 1]
        """
        X_t = X.T
        hid_layer1 = self.sigmoid((self.weights[0] @ X_t) + self.biases[0])
        hid_layer2 = self.sigmoid((self.weights[1] @ hid_layer1) + self.biases[1])
        pred_prob = self.sigmoid((self.weights[2] @ hid_layer2) + self.biases[2])

        return pred_prob

    def predict(self, X, desicion_thresh=0.5):
        """predict
        Compute the output of the neural network prediction for
        each sample in X, with the last layer's sigmoid activation
        providing an estimate of the target output between 0 and 1,
        then thresholding that prediction based on decision_thresh
        to produce a binary class prediction
            Input:
                X: A matrix of N samples of data [N x n_in]
                decision_threshold: threshold for the class confidence score
                                    of predict_proba for binarizing the output
            Output:
                y_hat: A vector of class predictions of either 0 or 1 [N x 1]"""
        pred_prob = self.predict_proba(X)
        pred_label = np.where(pred_prob >= desicion_thresh, 1, 0)
        return pred_label

    def compute_loss(self, X, y):
        """compute_loss
        Computes the current loss/cost function of the neural network
        based on the weights and the data input into this function.
        To do so, it runs the X data through the network to generate
        predictions, then compares it to the target variable y using
        the cost/loss function
            Input:
                X: A matrix of N samples of data [N x n_in]
                y: Target variable [N x 1]
            Output:
                loss: a scalar measure of loss/cost
        """
        y = y.flatten()
        y_hat = self.predict_proba(X)
        cost_i = (np.log(y_hat) * y) + ((1 - y) * np.log(1 - y_hat))
        avg_cost = -1 * np.mean(cost_i)
        return avg_cost

    def backpropagate(self, x, y):
        """backpropagate
        Backpropagate the error from one sample determining the gradients
        with respect to each of the weights in the network. The steps for
        this algorithm are:
            1. Run a forward pass of the model to get the activations
               Corresponding to x and get the loss functionof the model
               predictions compared to the target variable y
            2. Compute the deltas (see lecture notes) and values of the
               gradient with respect to each weight in each layer moving
               backwards through the network

            Input:
                x: A vector of 1 samples of data [n_in x 1]
                y: Target variable [scalar]
            Output:
                loss: a scalar measure of th loss/cost associated with x,y
                      and the current model weights
        """
        y_hat = self.forward_propagation(x)
        self.deltas[-1] = self.act_sig[-1] - y
        self.w_gradients[-1] = self.deltas[-1] @ (self.act_sig[-2].T)

        self.deltas[-2] = ((self.weights[-1].T) @ self.deltas[-1]) * (
            self.sigmoid_derivative(self.activations[-2])
        )
        self.w_gradients[-2] = self.deltas[-2] @ (self.act_sig[-3].T)

        self.deltas[-3] = ((self.weights[-2].T) @ self.deltas[-2]) * (
            self.sigmoid_derivative(self.activations[-3])
        )
        self.w_gradients[-3] = self.deltas[-3] @ (x.T)

        self.bias_grad = self.deltas.copy()

        cost_i = (np.log(y_hat) * y) + ((1 - y) * np.log(1 - y_hat))
        return -1 * cost_i

    def stochastic_gradient_descent_step(self):
        """stochastic_gradient_descent_step
        Using the gradient values computed by backpropagate, update each
        weight value of the model according to the familiar stochastic
        gradient descent update equation.

        Input: none
        Output: none
        """
        weights = np.array(self.weights)
        w_gradients = np.array(self.w_gradients)
        biases = np.array(self.biases)
        bias_grad = np.array(self.bias_grad)

        update_weights = self.lr * w_gradients
        update_biases = self.lr * bias_grad
        weights = weights - update_weights
        biases = biases - update_biases

        self.weights = list(weights)
        self.biases = list(biases)

    def fit(self, X, y, max_epochs=200):
        """fit
        Input:
            X: A matrix of N samples of data [N x n_in]
            y: Target variable [N x 1]
        Output:
            training_loss:   Vector of training loss values at the end of each epoch
            validation_loss: Vector of validation loss values at the end of each epoch
                             [optional output if get_validation_loss==True]
        """

        loss_for_epochs = []
        epochs = 0
        while epochs < max_epochs:
            data = np.column_stack((X, y)).copy()
            np.random.shuffle(data)
            y_train = data[:, -1]
            x_train = data[:, :-1].T
            epoch_loss_vec = []

            for i in range(x_train.shape[1]):
                sample_x = x_train[:, i].reshape(-1, 1)
                sample_y = y_train[i]
                loss = self.backpropagate(sample_x, sample_y)
                epoch_loss_vec.append(loss)
                self.stochastic_gradient_descent_step()

            avg_loss_epoch = np.mean(epoch_loss_vec)
            loss_for_epochs.append(avg_loss_epoch)
            epochs += 1

        return loss_for_epochs

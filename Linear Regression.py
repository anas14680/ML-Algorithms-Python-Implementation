import numpy as np


class MyLinearModel:
    def __init__(self):
        self.betas = None
        self.standard_errors = None
        self.resid = None
        self.R_squared = None

    def fit(self, X, Y, intercept=True):
        if intercept == True:

            a = X.shape[0]
            b = np.ones((a, 1))
            X_new = np.hstack((b, X))
        elif intercept == False:
            X_new = X
        assert isinstance(intercept, bool), "Intercept only takes boolean"

        X_t = X_new.transpose()
        prod = np.matmul(X_t, X_new)
        prod_inv = np.linalg.inv(prod)
        XY = X_t @ Y
        self.betas = prod_inv @ XY
        self.resid = Y - X_new @ self.betas
        avg_resid = float(sum(self.resid)) / len(self.resid)
        p = self.resid - avg_resid
        sigma_squared = np.dot(p[:, 0], p[:, 0]) / (len(self.resid) - 1)
        st_errors = sigma_squared * prod_inv
        standard_errors_squared = np.diag(st_errors)
        self.standard_errors = np.sqrt(standard_errors_squared)

        RSS = np.sum(self.resid ** 2)
        TSS = np.sum((Y - np.mean(Y)) ** 2)
        self.R_squared = 1 - (RSS / TSS)

    def residuals(self):
        return self.resid

    def standard_errors(self):
        return self.standard_errors

    def R_squared(self):
        return self.R_squared

    def predict(self, X):
        a = X.shape[0]
        b = np.ones((a, 1))
        X_new = np.hstack((b, X))

        if self.betas is None:
            return "train the model first"
        else:
            predicted = X_new @ self.betas
            return predicted

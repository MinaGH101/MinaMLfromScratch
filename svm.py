import numpy as np

########################## SVM for binary classification ######################

class SVM():

    def __init__(self, lr=0.001, lambda_param=0.01, iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.iters = iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_s, n_f = X.shape
        self.w = np.zeros(n_f)
        self.b = 0

        for _ in range(self.iters):
            for i, x_i in enumerate(X):
                cond = y_[i] * (np.dot(x_i, self.w) - self.b) >= 1
                if cond:
                    self.w -= self.lr * (2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param*self.w - np.dot(x_i, y_[i]))
                    self.b -= self.lr * y_[i]


    def predict(self, X):
        out = np.dot(X, self.w) - self.b
        return np.sign(out)

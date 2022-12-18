import numpy as np
import math


class LogRegressor():
    def __init__(self, lr=0.001, iters=1500):
        self.lr = lr
        self.iters= iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iters):
            linear_model = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T , (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        
    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_pred = self._sigmoid(linear_model)
        pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return pred_class

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
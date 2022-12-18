import numpy as np
from decision_tree import DecisionTree
from decision_tree_utils import bootstrap_data, most_common_label


class RandomForest():

    def __init__(self, n_t = 50, min_sample_split=5, max_depth=20, n_f=None):
        self.n_t = n_t
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_f = n_f


    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_t):
            t = DecisionTree(min_sample_split=self.min_sample_split, 
            max_depth=self.max_depth, n_f=self.n_f)
            X_sample, y_sample = bootstrap_data(X, y)
            t.fit(X_sample, y_sample)
            self.trees.append(t)

    def predivt(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        preds = np.swapaxes(preds, 0, 1)
        y_pred = [most_common_label(pred) for pred in preds]
        
        return np.array(y_pred)
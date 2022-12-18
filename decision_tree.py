import numpy as np
from decision_tree_utils import _common_class_calc, _calculate_variance_reduc, _information_gain, _split, mean

#################################### Tree #######################################


class Node():
    def __init__(self, features=None, threshold=None, left=None, right=None, * , value=None):
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree():
    def __init__(self, min_sample_split=5, max_depth=50, n_f = None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_f = n_f
        self.root = None
        self._impurity_calculation = None
        self._leaf_value = None

    def fit(self, X, y):
        self.n_features = X.shpae[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels ==1 or n_samples< self.min_sample_split):
            leaf_value = self._leaf_value
            return Node(value=leaf_value)

        feature_is = np.random.choice(n_features, self.n_f, replace=False)
        best_f, best_t = self._best_criteria(X, y, feature_is)
        left_is , right_is = self._split(X[: , best_f], best_t)

        left = self._grow_tree(X[left_is, :], y[left_is], depth+1)
        right = self._grow_tree(X[right_is, :], y[right_is], depth+1)

        return Node(best_f, best_t, left=left, right=right)

    def _best_criteria(self, X, y, feature_is):
        best_gain = -1
        split_i, split_threshold = None, None
        for feature_i in feature_is:
            X_col = X[: , feature_i]
            thresholds = np.unique(X_col)
            for t in thresholds:
                gain = self._impurity_calculation(X_col, t, y)

                if gain > best_gain:
                    best_gain = gain
                    split_i = feature_i
                    split_threshold = t

        return split_i, split_threshold


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])


    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_i] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


################################ Regression Tree ##################################


class DecisianTreeRegressor(DecisionTree):

    def fit(self, X, y):
        self._impurity_calculation = _calculate_variance_reduc
        self._leaf_value = mean



############################### Classification Tree #############################

class DecisianTreeClassifier(DecisionTree):

    def fit(self, X, y):
        self._impurity_calculation = _information_gain
        self._leaf_value = _common_class_calc
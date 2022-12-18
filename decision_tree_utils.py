import numpy as np
from collections import Counter

#####################################################################
####                            Utils                            ####
#####################################################################

def _split(X_col, t):
        left_is = np.argwhere(X_col <= t).flatten()
        right_is = np.argwhere(X_col > t).flatten()

        return left_is, right_is

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)

    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def variance(X):
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance

def _calculate_variance_reduc(X_col, y, t):
    parent_var = variance(y)

    left_is, right_is = _split(X_col, t)
    if len(left_is) == 0 or len(right_is) == 0 :
            return 0

    n = len(y)
    n_left, n_right = len(left_is), len(right_is)

    left_var = variance(y[left_is])
    right_var = variance(y[right_is])

    var_reduc = parent_var - ((n_left/n)*left_var + (n_right/n)*right_var)

    return var_reduc

def _common_class_calc(y):
    most_common = Counter(y).most_common(1)[0][0]
    return most_common

def _information_gain(self, X_col, t, y):
    parent_entropy = entropy(y)

    left_is, right_is = self._split(X_col, t)
    if len(left_is) == 0 or len(right_is) == 0 :
        return 0

    n = len(y)
    n_left, n_right = len(left_is), len(right_is)
    entropy_left, entropy_right = entropy(y[left_is]), entropy(y[right_is])
    child_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right

    info_gain = parent_entropy - child_entropy
    return info_gain


def mean(self, y):
    v = np.mean(y, axis=0)
    return v if len(v) > 1 else v[0]


def bootstrap_data(X, y):
    n = X.shape[0]
    i = np.random.choice(n, size=n, replace=True)
    return X[i], y[i]


def most_common_label(y):
    counter = Counter(y)
    mc = counter.most_common(1)[0][0]
    return mc

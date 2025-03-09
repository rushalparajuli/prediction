import numpy as np
from collections import Counter
from joblib import Parallel, delayed

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None, categorical_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)

        if n_samples == 0:
            return Node(value=0.0)

        parent_prob = self._class_probability(y)

        if (depth >= self.max_depth or len(unique_labels) == 1 or n_samples < self.min_samples_split):
            return Node(value=parent_prob)

        max_features = self.max_features or n_features
        max_features = min(max_features, n_features)
        best_feature, best_threshold = self._best_split(X, y, max_features)

        if best_feature is None:
            return Node(value=parent_prob)

        is_categorical = best_feature in self.categorical_features
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold, is_categorical)

        left_subtree = (
            self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
            if len(left_idx) > 0
            else Node(value=parent_prob)
        )
        right_subtree = (
            self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
            if len(right_idx) > 0
            else Node(value=parent_prob)
        )

        return Node(best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, max_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        features = np.random.choice(X.shape[1], max_features, replace=False)

        for feature in features:
            X_column = X[:, feature]
            is_categorical = feature in self.categorical_features

            if is_categorical:
                unique_values = np.unique(X_column)
                if len(unique_values) < 2:
                    continue
                for value in unique_values:
                    gain = self._information_gain(y, X_column, value, is_categorical=True)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feature
                        split_threshold = value
            else:
                unique_values = np.unique(X_column)
                if len(unique_values) < 2:
                    continue
                sorted_values = np.sort(unique_values)
                midpoints = (sorted_values[:-1] + sorted_values[1:]) / 2
                max_thresholds = 100
                if len(midpoints) > max_thresholds:
                    midpoints = np.random.choice(midpoints, max_thresholds, replace=False)
                for threshold in midpoints:
                    gain = self._information_gain(y, X_column, threshold, is_categorical=False)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feature
                        split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold, is_categorical=False):
        left_idx, right_idx = self._split(X_column, threshold, is_categorical)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_idx])
        right_entropy = self._entropy(y[right_idx])

        left_weight = len(left_idx) / n
        right_weight = len(right_idx) / n
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _split(self, X_column, threshold, is_categorical):
        if is_categorical:
            left_idx = np.where(X_column == threshold)[0]
            right_idx = np.where(X_column != threshold)[0]
        else:
            left_idx = np.where(X_column <= threshold)[0]
            right_idx = np.where(X_column > threshold)[0]
        return left_idx, right_idx

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _class_probability(self, y):
        counts = np.bincount(y)
        total = len(y)
        if total == 0:
            return 0.0
        prob = counts / total
        if len(prob) > 1:
            return prob[1]
        else:
            return 0.0

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if node.feature in self.categorical_features:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, 
                 max_features=None, n_jobs=None, categorical_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y) for _ in range(self.n_trees)
        )

    def _build_tree(self, X, y):
        X_sample, y_sample = self._bootstrap_sample(X, y)
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            categorical_features=self.categorical_features
        )
        tree.fit(X_sample, y_sample)
        return tree

    def _bootstrap_sample(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        sampled_indices = []
        for cls, count in zip(classes, counts):
            cls_indices = np.where(y == cls)[0]
            cls_sampled = np.random.choice(cls_indices, size=count, replace=True)
            sampled_indices.extend(cls_sampled)
        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)
        return X[sampled_indices], y[sampled_indices]

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X)
        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        avg_proba = np.mean(tree_preds, axis=0)
        return np.vstack((1 - avg_proba, avg_proba)).T

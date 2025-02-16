import numpy as np
from collections import Counter

# -------------------------
# Decision Tree (No Changes)
# -------------------------
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
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        # Convert X and y to NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)

        # Stop if no samples or other criteria met
        if n_samples == 0:
            return Node(value=0)  # Default value for empty node
        if (depth >= self.max_depth or len(unique_labels) == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Select best split
        max_features = self.max_features or n_features
        max_features = min(max_features, n_features)
        best_feature, best_threshold = self._best_split(X, y, max_features)

        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # Split data and handle empty subsets
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        
        # Use parent's majority class if child has no samples
        parent_common_label = self._most_common_label(y)
        left_subtree = (
            self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
            if len(left_idx) > 0
            else Node(value=parent_common_label)
        )
        right_subtree = (
            self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
            if len(right_idx) > 0
            else Node(value=parent_common_label)
        )

        return Node(best_feature, best_threshold, left_subtree, right_subtree)
    def _best_split(self, X, y, max_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        features = np.random.choice(X.shape[1], max_features, replace=False)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_idx])
        right_entropy = self._entropy(y[right_idx])

        n = len(y)
        left_weight = len(left_idx) / n
        right_weight = len(right_idx) / n
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _split(self, X_column, threshold):
        left_idx = np.where(X_column <= threshold)[0]
        right_idx = np.where(X_column > threshold)[0]
        return left_idx, right_idx

    def _entropy(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        # Convert X to NumPy array
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        # Ensure X and y are NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Convert X to NumPy array
        X = np.asarray(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._most_common_label(preds) for preds in tree_preds])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

import numpy as np
import scipy.spatial

class NearestNeighborDistanceMetric:
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "cosine":
            self.metric = self._cosine_distance
        else:
            raise ValueError("Invalid metric")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget and len(self.samples[target]) > self.budget:
                self.samples[target] = self.samples[target][-self.budget:]

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for row, target in enumerate(targets):
            cost_matrix[row, :] = self.metric(self.samples[target], features)
        return cost_matrix

    def _cosine_distance(self, a, b):
        return 1. - np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

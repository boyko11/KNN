import numpy as np


class KnnLearner:

    def __init__(self):
        self.k = 5
        self.training_records_feature_data = None
        self.training_records_labels = None

    def predict(self, feature_data):

        euclidean_distances = np.linalg.norm(self.training_records_feature_data - feature_data, axis=1)
        k_smallest_indices = np.argpartition(euclidean_distances, self.k + 1)[:self.k + 1]
        #+1 because we may try with an identical record

        k_smallest_distances = euclidean_distances[k_smallest_indices]
        k_nearest_labels = self.training_records_labels[k_smallest_indices]

        non_identical_record_indices = np.where(k_smallest_distances != 0)[0]
        k_smallest_distances = k_smallest_distances[non_identical_record_indices]
        k_nearest_labels = k_nearest_labels[non_identical_record_indices]

        unique_neighbor_labels = np.unique(k_nearest_labels)
        if unique_neighbor_labels.size == 1:
            return unique_neighbor_labels[0]

        k_smallest_distances_inverted = 1.0/k_smallest_distances
        k_nearest_neighbors_weights = k_smallest_distances_inverted / np.sum(k_smallest_distances_inverted)

        class_weights = {}
        for class_label in unique_neighbor_labels:
            neighbor_indices_for_class_labels = np.argwhere(k_nearest_labels == class_label)
            class_weights[class_label] = np.sum(k_nearest_neighbors_weights[neighbor_indices_for_class_labels])

        prediction = max(class_weights, key=class_weights.get)
        return prediction

    @staticmethod
    def calculate_cost(predictions, labels):
        return np.sum(np.abs(np.array(predictions) - labels))

    def train(self, feature_data, labels, k=5):

        self.training_records_feature_data = feature_data
        self.training_records_labels = labels
        self.k = k






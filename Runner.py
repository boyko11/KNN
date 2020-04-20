from service.data_service import DataService
import numpy as np
from knn_learner import KnnLearner


class Runner:

    def __init__(self, normalization_method='z'):
        self.knn_learner = None
        self.normalization_method = normalization_method

    def run(self):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        labels_data = data[:, 1]

        self.knn_learner = KnnLearner()

        normalized_data = DataService.normalize(data, method='z')
        normalized_feature_data = normalized_data[:, 2:]

        self.knn_learner.train(normalized_feature_data, labels_data, k=5)

        predictions = []
        for test_record in normalized_feature_data:
            prediction = self.knn_learner.predict(test_record)
            predictions.append(prediction)

        predictions = np.array(predictions)

        accuracy = 1 - np.sum(np.abs(predictions - labels_data)) / labels_data.shape[0]

        print("Accuracy: ", accuracy)

        positive_labels_count = np.count_nonzero(labels_data)
        negative_labels_count = labels_data.shape[0] - positive_labels_count
        positive_predictions_count = np.count_nonzero(predictions)
        negative_predictions_count = labels_data.shape[0] - positive_predictions_count

        print("Positive Labels, Positive Predictions: ", positive_labels_count, positive_predictions_count)
        print("Negative Labels, Negative Predictions: ", negative_labels_count, negative_predictions_count)

        labels_for_class1_predictions = labels_data[predictions == 1]
        true_positives_class1 = np.count_nonzero(labels_for_class1_predictions)
        false_negatives_class0 = labels_for_class1_predictions.shape[0] - true_positives_class1

        labels_for_class0_predictions = labels_data[predictions == 0]
        false_negatives_class1 = np.count_nonzero(labels_for_class0_predictions)
        true_positives_class0 = labels_for_class0_predictions.shape[0] - false_negatives_class1

        print('Class 1, true_positives, false_positives: ', true_positives_class1,
              positive_predictions_count - true_positives_class1)
        precision_class1 = np.around(true_positives_class1/positive_predictions_count, 3)
        recall_class1 = np.around(true_positives_class1 / (true_positives_class1 + false_negatives_class1), 3)
        class1_f1_score = np.around(2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1), 3)

        print('Class 0, true_positives, false_positives: ', true_positives_class0,
              negative_predictions_count - true_positives_class0)
        precision_class0 = np.around(true_positives_class0/negative_predictions_count, 3)
        recall_class0 = np.around(true_positives_class0 / (true_positives_class0 + false_negatives_class0), 3)
        class0_f1_score = np.around(2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0), 3)

        print('precision class1: ', precision_class1)
        print('recall class1: ', recall_class1)
        print('f1 score class1: ', class1_f1_score)
        print('precision class0: ', precision_class0)
        print('recall class0: ', recall_class0)
        print('f1 score class0: ', class0_f1_score)

        cost = self.knn_learner.calculate_cost(predictions, labels_data)
        print("Final Cost: ", cost)

        self.print_error_stats(data, labels_data, predictions)

    @staticmethod
    def print_error_stats(data, labels_data, predictions):
        record_ids = data[:, 0].flatten()
        np.set_printoptions(suppress=True)
        # | Record ID | Label | Prediction Error |
        for i in range(labels_data.shape[0]):
            record_id = record_ids[i]
            label = 'Malignant' if labels_data[i] == 1 else 'Benign'
            prediction_error = np.abs(labels_data[i] - predictions[i])
            print('|{0}|{1}|{2}|'.format(int(record_id), label, prediction_error))


if __name__ == "__main__":

    Runner(normalization_method='z').run()

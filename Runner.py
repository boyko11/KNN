from service.data_service import DataService
import numpy as np
from knn_learner import KnnLearner
from service.report_service import ReportService


class Runner:

    def __init__(self, normalization_method='z'):
        self.knn_learner = None
        self.normalization_method = normalization_method
        self.report_service = ReportService()

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

        self.report_service.report(data, predictions, labels_data, self.knn_learner)



if __name__ == "__main__":

    Runner(normalization_method='z').run()

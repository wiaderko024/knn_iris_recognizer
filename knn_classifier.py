from operator import itemgetter
import csv
import os


class KnnClassifier:
    def __init__(self, k, data_filename, test_filename):
        self.k = k
        self.data = self.read_data(
            f'{os.getcwd()}/data/{data_filename}'
        )
        self.test_data = self.read_data(
            f'{os.getcwd()}/data/{test_filename}'
        )

    @staticmethod
    def read_data(path):
        data = []
        with open(path) as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
        return data

    @staticmethod
    def knn(vector, data, k):
        results = []
        classes_counter = {}

        for row in data:
            classes_counter[row[len(row) - 1]] = 0

            result = 0
            for num in range(0, len(row) - 1):
                result += (float(vector[num]) - float(row[num])) ** 2

            results.append([row[len(row) - 1], result])

        results.sort(key=itemgetter(1))

        for i in range(int(k)):
            classes_counter[results[int(k)][0]] += 1

        return max(classes_counter, key=classes_counter.get)

    def train(self):
        good_results = 0

        for data in self.test_data:
            vector = []
            for num in range(len(data) - 1):
                vector.append(data[num])

            result = self.knn(vector, self.data, self.k)

            if result == data[len(data) - 1]:
                good_results += 1

        print('Accuracy of classification: ' + str(good_results / len(self.test_data))
              + ' (' + str(good_results) + '/' + str(len(self.test_data)) + ')')

    def classify(self, vector):
        if len(vector) == len(self.data[0]) - 1:
            print(self.knn(vector, self.data, self.k))
        else:
            print(f'Give a {len(self.data[0]) - 1} dimensional vector')

import numpy as np


class KNearestNeighbor():
    def __init__(self, k=3, distance="l2"):
        self.k = k
        self.distance = distance

    def train(self, X, y):
        self.x_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        d = 0.0
        for i in range(len(x1)):
            d += (x1[i] - x2[i])**2
        return np.sqrt(d)

    def manhathan_distance(self, x1, x2):
        d = 0.0
        for i in range(len(x1)):
            d += np.abs(x1[i] - x2[i])
        return d

    def get_k_neighbor(self, x_test):
        distance = []
        neighbors = []
        if self.distance == "l2":
            for i in range(len(self.x_train)):
                dist = self.euclidean_distance(self.x_train[i], x_test)
                distance.append((i, dist))
            distance.sort(key=lambda x: x[1])
        elif self.distance == "l1":
            for i in range(len(self.x_train)):
                dist = self.manhathan_distance(self.x_train[i], x_test)
                distance.append((i, dist))
            distance.sort(key=lambda x: x[1])

        for i in range(self.k):
            neighbors.append(distance[i][0])

        return neighbors

    def predict_one(self, x_test):
        neighbor = self.get_k_neighbor(x_test)
        class_ = {}

        for i in neighbor:
            if self.y_train[i] in class_:
                class_[self.y_train[i]] += 1
            else:
                class_[self.y_train[i]] = 1

        return max(class_, key=class_.get)

    def predict(self, x_test):
        output = []
        for i in x_test:
            output.append(self.predict_one(i))
        return output

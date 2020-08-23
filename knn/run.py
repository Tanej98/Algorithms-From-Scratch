from knn import KNearestNeighbor

if __name__ == "__main__":
    x = [[2, 2],
         [1, 2],
         [3, 4],
         [1, 1],
         [3, 3],
         [7, 2],
         [5, 2],
         [6, 1],
         [8, -0],
         [7, 3]]

    target = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    knn = KNearestNeighbor(3)

    knn.train(x, target)

    output = knn.predict([[8, 3], [1, 1]])

    print(output)

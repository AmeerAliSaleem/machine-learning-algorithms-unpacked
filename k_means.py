import numpy as np

# An implementation of the K-Means algorithm from scratch.
# Thanks to Emma Ding's YT video for helping me write the code for this class: https://youtu.be/uLs-EYUpGAw?si=N-uPPx6cty5af7Ey
class KMeans:
    def __init__(self, k=3, max_iter=100, tol=0.001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def euclidean_distance(self, point1, point2):
        """
        Computes the Euclidean distance between the pair of input points.
        """

        return np.linalg.norm(point1-point2)
    
    def initialise_centroids(self, data):
        """
        Initialises the centroids' coordinates randomly within the data range.
        If we just took k random points from the data, the initialisation will be biased
        if the data is not uniformly spread across the data space.
        """

        # separate x and y coordinates
        data_x = np.array([point[0] for point in data])
        data_y = np.array([point[1] for point in data])

        x_min = np.min(data_x)
        x_max = np.max(data_x)
        y_min = np.min(data_y)
        y_max = np.max(data_y)

        data_min = np.array([x_min, y_min])
        data_max = np.array([x_max, y_max])

        # generate k random centroids within the data range
        centroids = [data_min + np.random.uniform() * (data_max - data_min) for _ in range(self.k)]

        return centroids
    
    def get_labels(self, data, centroids):
        """
        Assigns each data point to the nearest centroid.
        """

        labels = []

        for point in data:
            min_distance = np.inf
            label = None
            for i, centroid in enumerate(centroids):
                # aim is to store the index of the centroid that the current point is closest to
                current_distance = self.euclidean_distance(point, centroid)
                if current_distance < min_distance:
                    min_distance = current_distance
                    label = i
            labels.append(label)

        return labels

    def update_centroids(self, data, labels):
        """
        Updates the centroids' coordinates based on the current labels
        """

        new_centroids = []

        for i in range(self.k):
            # calculate the mean of the data points that belong to the current cluster
            new_centroid = np.mean([data[j] for j in range(len(data)) if labels[j] == i])

            new_centroids.append(new_centroid)

        return new_centroids

    def stopping_criteria(self, old_centroids, new_centroids, current_iter):
        """
        Checks whether the algorithm should terminate at the current iteration.
        This happens if either: the total centroid deviations do not exceed the tolerance level;
        or the maximum number of iterations has been reached.
        """
        
        total_deviation = 0
        for i in range(self.k):
            total_deviation += self.euclidean_distance(old_centroids[i], new_centroids[i])
        
        if total_deviation < self.tol or current_iter >= self.max_iter:
            return True
        else:
            return False

    
    def fit(self, data):
        """
        Applies the K-Means algorithm to the input data.
        """

        # initialise centroids randomly within the data range
        centroids = self.initialise_centroids(data)

        # main loop of algorithm
        current_iter = 0
        while True:
            old_centroids = centroids
            labels = self.get_labels(data, centroids)
            centroids = self.update_centroids(data, labels)

            current_iter += 1

            # check stopping criteria
            stopping_criteria = self.stopping_criteria(old_centroids, centroids, current_iter)
            if stopping_criteria:
                break

        # store labels and centroids
        self.labels = labels
        self.centroids = centroids
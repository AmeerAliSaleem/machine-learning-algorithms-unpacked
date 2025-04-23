import numpy as np

# An implementation of the K-Means algorithm from scratch.
# Thanks to Emma Ding's YT video for helping me write the code for this class: https://youtu.be/uLs-EYUpGAw?si=N-uPPx6cty5af7Ey
class KMeans:
    """
    A class for performing K-Means clustering on a dataset. The data is expected to be in the form
    of a list of NumPy ndarrays, where each ndarray represents the coordinates of a data point.
    """
    
    def __init__(self, k=3, max_iter=100, tol=0.001):
        """
        Parameters
        ----------
        k : int, optional
            The number of clusters to form (default is 3).
        max_iter : int, optional
            The maximum number of iterations to run the algorithm for (default is 100).
        tol : float, optional
            The tolerance level for the stopping criteria (default is 0.001).
        """

        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def euclidean_distance(self, point1, point2):
        """
        Computes the Euclidean distance between the pair of input points.

        Parameters
        ----------
        point1 : np.ndarray
            The first point data point
        point2 : np.ndarray
            The second data point.

        Returns
        ----------
        distance : float
            The Euclidean distance between the two points.
        """

        distance = np.linalg.norm(point1-point2)
        return distance
    
    def initialise_centroids(self, data):
        """
        Initialises the centroids' coordinates randomly within the data range.
        If we just took k random points from the data, the initialisation will be biased
        if the data is not uniformly spread across the data space, hence why we instead
        implement this 'bounding box' method.

        Parameters
        ----------
        data : list
            List of NumPy ndarrays. Each array represents the coordinates of a data point.

        Returns
        ----------
        centroids : list
            List of NumPy ndarrays. Each array represents the coordinates of a randomly initialised centroid.
        """

        # group together coordinates per dimension
        stack = np.stack(data, axis=1)

        # create arrays to store the minimum/maximum values in each coordinate dimension
        data_min = np.min(stack, axis=1)
        data_max = np.max(stack, axis=1)

        # generate k random centroids within the data range
        # (this is more robust than just taking k random points from the data)
        centroids = [data_min + np.random.uniform(size=len(stack))*(data_max - data_min) for _ in range(self.k)]

        return centroids
    
    def get_labels(self, data, centroids):
        """
        Assigns each data point to the nearest centroid.

        Parameters
        ----------
        data : list
            List of NumPy ndarrays. Each array represents the coordinates of a data point.
        centroids : list
            List of NumPy ndarrays. Each array represents the coordinates of a centroid.

        Returns
        ----------
        labels : list
            List of integers. Each integer represents the cluster whose centroid each data point is closest to.
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
        Updates the centroids' coordinates based on the current labels.

        Parameters
        ----------
        data : list
            List of NumPy ndarrays. Each array represents the coordinates of a data point.
        labels : list
            List of integers. Each integer represents the cluster whose centroid each data point is closest to.

        Returns
        ----------
        new_centroids : list
            List of NumPy ndarrays. Each array represents the updated coordinates of a centroid.
        """

        new_centroids = []

        for i in range(self.k):
            # calculate the mean of the data points that belong to the current cluster
            data_in_cluster = [data[j] for j in range(len(data)) if labels[j] == i]
            data_stack = np.stack(data_in_cluster, axis=1)
            new_centroid = np.mean(data_stack, axis=1)

            new_centroids.append(new_centroid)

        return new_centroids

    def stopping_criteria(self, old_centroids, new_centroids, current_iter):
        """
        Checks whether the algorithm should terminate at the current iteration.
        This happens if either: the total centroid deviations do not exceed the tolerance level;
        or the maximum number of iterations has been reached.

        Parameters
        ----------
        old_centroids : list
            List of NumPy ndarrays. Each array represents the previous coordinates of a centroid.
        new_centroids : list
            List of NumPy ndarrays. Each array represents the new coordinates of a centroid.
        current_iter : int
            The current iteration of the algorithm.

        Returns
        ----------
        boolean representing whether the algorithm should terminate (True) or not (False).
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
        Applies the K-Means algorithm to the input data. The cluster centroids and data point labels
        are stored in lists. The i-th element of each list corresponds to the labellings/centroid allocations 
        at the i-th iteration of the algorithm.

        Parameters
        ----------
        data : np.ndarray
            The input data to perform the clustering on.

        Returns
        ----------
        None, although the following attributes are set:
        self.labels : list
            The labels assigned to each data point. Each element in the list corresponds to the 
            labellings at the i-th iteration of the algorithm.
        self.centroids : list
            The coordinates of the centroids. Each element in the list corresponds to the 
            centroid allocations at the i-th iteration of the algorithm.
        """

        # initialise output lists
        i = 0
        labels, centroids = [], []

        # initialise centroids randomly within the data range
        centroids.append(self.initialise_centroids(data))

        # initialise labels based on the centroid initialisations
        labels.append(self.get_labels(data, centroids[0]))

        # main loop of algorithm
        while True:
            i += 1

            # update centroids and labels
            old_centroids = centroids[i-1]
            new_centroids = self.update_centroids(data, labels[i-1])
            new_labels = self.get_labels(data, new_centroids)

            # store new centroids and new labels
            centroids.append(new_centroids)
            labels.append(new_labels)
            

            # check stopping criteria
            stopping_criteria = self.stopping_criteria(old_centroids, new_centroids, i)
            if stopping_criteria:
                break

        # store lists for labels and centroids
        self.labels = labels
        self.centroids = centroids
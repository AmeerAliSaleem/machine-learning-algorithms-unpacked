import numpy as np

class dbscan:
    """
    A class for performing the DBSCAN algorithm on a dataset. The data is expected to be in the form
    of a list of NumPy ndarrays, where each ndarray represents the coordinates of a point.

    Attributes:
        eps : float
            The distance beneath which a pair of points are considered to be neighbours.
        minPoints : int
            The minimum number of neighbours a point must have to be considered a core point.
        history : dict
            The history of clustering assignments. Useful for Manim animations.
        labels : list
            The list containing the cluster labellings for each point.
            The i-th element contains the cluster index of the i-th point.
    """
    def __init__(self, eps=1, minPoints=5):
        """
        Parameters
        ----------
        eps : float, optional
            The distance beneath which a pair of points are considered to be neighbours.
        minPoints : int, optional
            The minimum number of neighbours a point must have to be considered a core point.
        """
        self.eps = eps
        self.minPoints = minPoints
        self.history = {}
    

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


    def neighbours(self, current_node, data):
        """
        Finds the indices of all the neighbours of {current_node} that are at most {eps}.

        Parameters
        ----------
        current_node : np.ndarray
            The data point to find the neighbours of.
        data : list
            The list of data points.

        Returns
        ----------
        closest_neighbours_indices : list
            A list containing the indices of all the neighbours of {current_node} that are at most {eps} away.
        """

        closest_neighbours = [(point, self.euclidean_distance(point, current_node)) for point in data]

        # remove neighbours that are not close enough
        closest_neighbours_indices = [index for index, val in enumerate(closest_neighbours) if val[1] <= self.eps]

        return closest_neighbours_indices


    def core_points(self, data):
        """
        Builds a list whose elements correspond to the indices of core points.

        Parameters
        ----------
        data : list
            The list of data points.

        Returns
        ----------
        core_points : list
            A list containing the indices of all the core points in the data.
        """

        core_points = []

        for i, point in enumerate(data):
            neighbours = self.neighbours(point, data)

            if len(neighbours) >= self.minPoints:
                core_points.append(i)
        
        return core_points


    def expand_cluster(self, data, labels, current_point_index, neighbours_indices, current_cluster_index):
        """
        Expands the current cluster starting from the data point with index {current_point_index}.

        Parameters
        ----------
        data : list
            The list of data points.
        labels : list
            The list containing the cluster labellings for each point. The i-th element contains the cluster index
            of the i-th point.
        current_point_index : int
            The index of the point at which the current cluster is being expanded from.
        neighbours_indices : list
            A list containing the indices of all the neighbours of the current data point.
        current_cluster_index : int
            The index of the cluster that is currently being expanded.

        Returns
        ----------
        labels : list
            The clustering labelling list that was given as input.
        """

        # store the cluster assignment for the current point
        labels[current_point_index] = current_cluster_index

        # list to keep track of the order in which clusters are assigned
        cluster_assigned = [current_point_index]
        
        i = 0
        visited = set(neighbours_indices)

        while i < len(neighbours_indices):
            neighbour_index = neighbours_indices[i]

            if labels[neighbour_index] == -1:
                # non-core point that is a neigbour to the current core point:
                # this non-core point simply joins the current cluster
                labels[neighbour_index] = current_cluster_index

                cluster_assigned.append(neighbour_index)
            elif labels[neighbour_index] == 0:
                # unassigned core point: the current cluster expands to this core point's neighbours
                labels[neighbour_index] = current_cluster_index
                cluster_assigned.append(neighbour_index)
                new_neighbours = self.neighbours(data[neighbour_index], data)
                if len(new_neighbours) >= self.minPoints:
                    # neighbours_indices.extend(new_neighbours)
                    for new_neighbour in new_neighbours:
                        if new_neighbour not in visited:
                            neighbours_indices.append(new_neighbour)
                            visited.add(new_neighbour)

            i += 1
        
        self.history[current_cluster_index] = cluster_assigned

        return labels


    def fit(self, data):
        """
        Applies the DBSCAN algorithm to cluster the data.
        If a random starting point for the algorithm is desired, then the data must first be shuffled.

        Parameters
        ----------
        data : list
            The list of data points.

        Returns
        ----------
        None, but stores the cluster allocations in the `labels` list of the class.
        """

        # instantiate output list
        labels = [0]*len(data)

        cluster_index = 0

        # store index values of all core points
        core_points = self.core_points(data)

        for current_point_index in range(len(data)):
            # go to the next point if the current point has already been classified/ignored
            if labels[current_point_index] != 0:
                continue

            if current_point_index in core_points:
                cluster_index += 1
                neighbours_indices = self.neighbours(data[current_point_index], data)
                labels = self.expand_cluster(data, labels, current_point_index, neighbours_indices, cluster_index)
            else:
                # non-core points are assigned the label '-1' for now
                # if this non-core point is an outlier, its label will stay as '-1'
                labels[current_point_index] = -1

        self.labels = labels     
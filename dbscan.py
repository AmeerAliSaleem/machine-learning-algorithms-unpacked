import numpy as np

class dbscan:
    def __init__(self, eps=1, minPoints=5):
        self.eps = eps
        self.minPoints = minPoints
    

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
        Finds the indices of all the neighbours of {currentnode} that are at most {eps} away.
        """

        closest_neighbours = [(point, self.euclidean_distance(point, current_node)) for point in data if not np.array_equal(point, current_node)]

        # sort ascending order of distance
        closest_neighbours.sort(key = lambda x: x[1])

        # remove neighbours that are not close enough
        closest_neighbours_indices = [index for index, val in enumerate(closest_neighbours) if val[1] <= self.eps]

        return closest_neighbours_indices


    def corePoints(self, data):
        """
        Builds a list whose elements correspond to the indices of core points.
        """

        core_points = []

        for i, point in enumerate(data):
            neighbours = self.neighbours(point, data)

            if len(neighbours) >= self.minPoints:
                core_points.append(i)
        
        return core_points


    def expandCluster(self, data, labels, current_point_index, neighbours_indices, current_cluster_index):
        """
        Expands the current cluster starting from the data point with index {current_point_index}.
        """

        labels[current_point_index] = current_cluster_index
        i = 0
        visited = set(neighbours_indices)

        while i < len(neighbours_indices):
            neighbour = neighbours_indices[i]

            if labels[neighbour] == -1:
                # non-core point: simply joins the current cluster
                labels[neighbour] = current_cluster_index
            elif labels[neighbour] == 0:
                # unassigned core point: the current cluster expands to this core point's neighbours
                labels[neighbour] = current_cluster_index
                # new_neighbours = self.neighbours(data[current_point_index], data)
                new_neighbours = self.neighbours(data[neighbour], data)
                # any core point should have its neighbours included in the current cluster
                if len(new_neighbours) >= self.minPoints:
                    neighbours_indices.extend(new_neighbours)

            i += 1
        
        return labels


    def fit(self, data):
        """
        Applies the DBSCAN algorithm to cluster the data.
        """

        # instantiate output list
        labels = [0]*len(data)

        cluster_index = 0

        # store index values of all core points
        core_points = self.corePoints(data)

        for current_point_index in range(len(data)):
            # go to the next point if the current point has already been classified/ignored
            if labels[current_point_index] != 0:
                continue

            if current_point_index in core_points:
                cluster_index += 1
                neighbours_indices = self.neighbours(data[current_point_index], data)
                labels = self.expandCluster(data, labels, current_point_index, neighbours_indices, cluster_index)
            else:
                labels[current_point_index] = -1

        self.labels = labels

        # # step 1: create dictionaries to store the information about the neighbours of all the points
        # core_dict = {}
        # non_core_dict = {}

        # for point in data:
        #     current_node_neighbours = self.nodeNeighbours(point)
        #     if len(current_node_neighbours) >= self.minPoints:
        #         core_dict[point] = self.nodeNeighbours(point)
        #     else:
        #         non_core_dict[point] = self.nodeNeighbours(point)

        # # step 2: 
        # clusters = {}
        # cluster_index = 0

        # # repeat the following as long as there are still core points that haven't been assigned a cluster
        # while core_dict:
        #     clusters[cluster_index] = []

        #     random_core_point = np.random.choice(list(core_dict.keys()))

        #     clusters[cluster_index].append(random_core_point)

        #     # add each core neighbour of random_core_point to the same cluster
        #     for neighbour in core_dict[random_core_point]:
        #         clusters[cluster_index].append(neighbour)

        #     core_dict.pop(random_core_point)

        #     cluster_index += 1        
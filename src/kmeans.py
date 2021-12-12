# Helper functions for the k-means algorithm
def l2_dist(pair):
    """
    Function used to calculate the l2 distance
    """
    d = (np.array(pair[0]) - np.array(pair[1]))
    return (pair, np.linalg.norm(d))

def filter_min(item):
    """
    Find the nearest cluster for a data point
    """
    key = item[0]
    value = item[1]
    min_val = value[0][1]
    new_val = []
    for i in value:
        if(i[1] == min_val):
            new_val.append(i)
    return (key, new_val)

def remap_for_centroid(pair):
    """
    Regroup data points into their clusters
    """
    key = pair[0]
    val = pair[1]
    new_key = val[0][0]
    new_val = []
    for i in val:
        if(i[0] == new_key):
            new_val.append((key, i[1]))
    return (new_key, new_val)

def get_mean_of_points(pair):
    """
    Calculate the new means for all the centroids
    """
    values = pair[1]
    val = []
    for i in values:
        val.append((i[0]))
    val = np.array(val)
    return (tuple(np.mean(val, axis=0)), values)

def get_total_cost(pair):
    """
    Calculates the cost function
    """
    cluster = pair[0]
    data_points = pair[1]
    total_cost = 0
    for x in data_points:
        point = x[0]
        d = np.linalg.norm(np.array(point) - np.array(cluster))**2
        total_cost += d
    return (cluster, total_cost)

# K-means function
def k_means(data, initial, max_iters=20):
    """
    K-means algorithm for Spark RDD's. This function takes in an RDD and an initial centroid set.

    :param data: The datapoints as an RDD object.
    :param initial: The initial centroids to start with. Must also be an RDD object.
    :param max_iters: The stopping criterion for the k-means. By default it will stop in 20 iterations.
    :returns: The cost function at each iteration.
    """
    total_cost_by_iteration = []
    centroids = initial
    for i in range(max_iters):
        # find the cluster closest to x
        # Get the cartesian
        cartesian = data.cartesian(centroids)
        # Calculate the distance for every point in data to a centroid
        distance = cartesian.map(l2_dist).map(lambda p: (p[0][0], (p[0][1], p[1])))
        # Group the distances by the data point
        grouped = distance.groupByKey().mapValues(list).mapValues(lambda p: sorted(p, key=lambda tup: (tup[1])))
        # Get the shortest distance to cluster only
        grouped_reduced = grouped.map(filter_min)

        # Group into their clusters
        centroid_map = grouped_reduced.map(remap_for_centroid)
        centroid_group = centroid_map.groupByKey().mapValues(list).mapValues(lambda t: [item for sublist in t for item in sublist])

        # Set the new centroids
        new_centroid = centroid_group.map(get_mean_of_points)
        centroids = new_centroid.map(lambda p: p[0])

        # Calculate the cost
        centroid_cost = centroid_group.map(get_total_cost)
        total_cost = centroid_cost.values().sum()
        total_cost_by_iteration.append(total_cost)
    return total_cost_by_iteration

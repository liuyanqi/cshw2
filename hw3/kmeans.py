"""
    This is a file you will have to fill in.

    It contains helper functions required by K-means method via iterative improvement

"""
import random
import numpy as np

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids

    :param k: number of cluster centroids, an int
    :param inputs: a 2D Python list, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroid_idx = []
    for i in range(k):
        centroid_idx.append(random.randint(0,len(inputs)))

    centroid = []
    for idx in centroid_idx:
        centroid.append(inputs[idx])

    return centroid


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance

    :param inputs: inputs of data, a 2D Python list
    :param centroids: a Numpy array of k current centroids
    :return: a Python list of centroid indices, one for each row of the inputs
    """
    # TODO
    centroid_idx = []

    for ip in inputs:
        min_dist = 10000000
        min_cent_idx = 0
        for idx, cent in enumerate(centroids):
            dist = np.linalg.norm(ip - cent)
            if dist < min_dist:
                min_dist = dist
                min_cent_idx = idx
        centroid_idx.append(min_cent_idx)
    return centroid_idx


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster - the average of all data points in the cluster

    :param inputs: inputs of data, a 2D Python list
    :param indices: a Python list of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    ##QUESTION: why k is needed

    centroid = np.zeros((k, len(inputs[0])))
    centroid_count = np.zeros(k)

    for idx, ip in enumerate(inputs):
        centroid[indices[idx]] += ip
        centroid_count[indices[idx]] += 1

    for idx in range(len(centroid)):
        if centroid_count[idx] != 0:
            centroid[idx] = centroid[idx] / centroid_count[idx]
    return centroid



def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    Use init_centroids, assign_step, and update_step!
    The only computation that should occur within this function is checking 
    for convergence - everything else should be handled by helpers

    :param inputs: inputs of data, a 2D Python list
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    iteration = 0
    error = 1000
    k_centriod = init_centroids(k, inputs)
    while iteration < max_iter and error > tol:
        input_cent = assign_step(inputs, k_centriod)
        k_centriod_new = update_step(inputs, input_cent, k)
        error = np.linalg.norm(k_centriod_new - k_centriod)
        # print("iteration: ", iteration, "error: ", error)
        k_centriod = k_centriod_new
        iteration += 1

    return k_centriod

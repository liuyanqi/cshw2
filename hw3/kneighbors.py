"""
    This is a file you will have to fill in.

    It contains helper functions required by K Nearest Neighbors method
    Note: get_neighbors_indices and get_response are designed to be called in sequence by models.py
"""
import numpy as np
import operator

def euclidean_distance(input1, input2):
    """
    Compute the euclidean distance between input1 and input2

    :param input1: input data point 1, a Python list
    :param input2: input data point 2, a Python list
    :return: euclidean distance between input1 and input2, a float number
    """
    # TODO

    # Recall the definition of Euclidean distance for two vectors x,y
    # d = sqrt(sum(x_i - y_i)^2)
    return np.linalg.norm(input1 - input2) 


def get_neighbors_indices(training_inputs, test_instance, k):
    """
    Get the indices of k closest neighbors to testInstance using Euclidean Distance
    Use the euclidean_distance helper function

    :param training_inputs: inputs of training data, a 2D Python list
    :param test_instance: an instance input of test data, a Python list
    :param k: number of neighbors used, an int
    :return: a Python list of indices of k closest neighbors from the training set for a given test instance
    """
    # TODO
    dist = {}
    for idx, t_ip in enumerate(training_inputs):
        d = euclidean_distance(t_ip, test_instance)
        dist[d] = idx

    top_k_index = []
    for key in sorted(dist)[:k]:
        top_k_index.append(dist[key])
    return top_k_index

def get_response(training_labels, neighbor_indices):
    """
    Get the most commonly voted response from a number of neighbors

    :param training_labels: labels of training data, a Python list
    :param neighbor_indices: a Python list of indices of k closest neighbors from the training data
    :return: the class/label with the highest vote, an int
    """
    # TODO]
    label_count = np.zeros(int(np.max(training_labels))+1)
    for idx in neighbor_indices:
        label_count[int(training_labels[idx])] +=1
        # print(training_labels[idx])

    return np.argmax(label_count)

import numpy as np
import requests as rq
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib.pyplot as plt

def parse_data(text_content):
    """
    Parses the given text_content and returns a numpy array.

    :param text_content: the string containing the data to be parsed
    :return: a numpy array representing the parsed data
    """
    lines = text_content.split('\n')
    data = np.array([[float(num) for num in line.split()] for line in lines if line.strip() != ''])
    return data

def distance_function(a, b):
    """
    Calculates the Euclidean distance between two numeric arrays `a` and `b`.

    :param a: The input array `a`.
    :param b: The input array `b`.
    :return: The Euclidean distance between `a` and `b`.

    """
    return np.sqrt(np.sum((a - b) ** 2))

def sum_of_squared_errors(data, centroids, partition):
    """
    Calculate the sum of squared errors for a given dataset and clustering.

    :param data: List of data points.
    :param centroids: List of centroid points.
    :param partition: List of assigned centroid indices for each data point.
    :return: Sum of squared errors.
    """
    total_error = 0
    for point, centroid_index in zip(data, partition):
        centroid = centroids[centroid_index]
        total_error += distance_function(point, centroid) ** 2
    return total_error

def dummy_cluster(data, k):
    """
    :param data: The data points to be clustered.
    :param k: The number of clusters to be created.
    :return: A tuple containing the centroids of the clusters and the partition of data points into clusters.
    """
    n_points = len(data)
    centroids = data[np.random.choice(n_points, k, replace=False)]
    partition = np.random.randint(0, k, size=n_points)
    return centroids, partition

def write_output(centroids, partition, centroid_file='centroid.txt', partition_file='partition.txt'):
    """
    Write the centroids and partition data to files.

    :param centroids: The centroids array.
    :param partition: The partition array.
    :param centroid_file: The file name to save the centroids data. Default is 'centroid.txt'.
    :param partition_file: The file name to save the partition data. Default is 'partition.txt'.
    :return: None.
    """
    np.savetxt(centroid_file, centroids, fmt="%0.1f")
    with open(partition_file, 'w') as f:
        for p in partition:
            f.write(f"{p}\n")

def optimal_partition(data, centroids):
    """

    :param data: numpy array of shape (N, D) representing the data points to be assigned to partitions.
    :param centroids: numpy array of shape (M, D) representing the centroids of the partitions.
    :return: numpy array of shape (N,) containing the index of the partition to which each data point is assigned.

    This method takes in a set of data points and a set of centroids and assigns each data point to the nearest centroid based on Euclidean distance. It uses a KDTree for efficient nearest
    * neighbor search.

    """
    tree = cKDTree(centroids)
    partition = tree.query(data)[1]
    return partition

def read_data_from_url():
    """
    Reads data from a URL and returns it.

    :return: The data retrieved from the URL.
    """
    try:
        file = rq.get('https://cs.uef.fi/sipu/datasets/s1.txt')
        file.raise_for_status()
    except Exception as e:
        print(f"Error with file url: {e}")
        quit()

    data = parse_data(file.text)
    return data

def fetch_new_centroids(data, partition, k):
    """
    Fetches the new centroids based on the given data, partition, and number of clusters.

    :param data: The data points used to calculate the new centroids.
    :type data: numpy.ndarray

    :param partition: The partition of each data point, indicating which cluster it belongs to.
    :type partition: list

    :param k: The number of clusters.
    :type k: int

    :return: The new centroids.
    :rtype: numpy.ndarray
    """
    new_centroids = []
    for i in range(k):
        points = [point for point, centroid_index in zip(data, partition) if centroid_index == i]
        centroid = np.mean(points, axis=0)
        new_centroids.append(centroid)
    return np.array(new_centroids)

def kmeans(data, k):
    """
    Perform k-means clustering on the given data.

    :param data: The input data to cluster.
    :param k: The number of clusters to create.
    :return: A tuple containing the final centroids, partition, and activity tracker.

    The method initializes the centroids and partition using the `dummy_cluster` method. Then, it enters a loop
    where it calculates a new partition and centroids using the `optimal_partition` and `fetch_new_centroids` methods.
    It also calculates the sum of squared errors (SSE) and tracks the number of active centroids.
    The loop continues until convergence is reached, where the new centroids are the same as the previous centroids.
    Finally, the method returns the final centroids, partition, and activity tracker.

    """
    centroids, partition = dummy_cluster(data, k)
    activity_tracker = []

    while True:
        new_partition = optimal_partition(data, centroids)
        new_centroids = fetch_new_centroids(data, new_partition, k)

        # Calculate SSE and number of active centroids
        sse = sum_of_squared_errors(data, new_centroids, new_partition)
        active_centroids = np.unique(new_partition)
        active_centroids_count = len(active_centroids)
        active_centroids_percent = (active_centroids_count / k) * 100

        # Track the values
        activity_tracker.append((sse, active_centroids_count, active_centroids_percent))

        # Check for convergence
        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids
        partition = new_partition

    return centroids, partition, activity_tracker

def main():
    """
    Main method that executes the k-means clustering algorithm on a dataset and visualizes the results.

    :return: None
    """
    data = read_data_from_url()
    k = int(input("Enter number of clusters: "))
    centroids, partition, activity_tracker = kmeans(data, k)

    # Convert the activity tracker to a DataFrame for easy manipulation
    activity_df = pd.DataFrame(activity_tracker, columns=['SSE', 'No. of active centroids', 'Percent of active centroids'])

    print(activity_df)

    # Plotting SSE values
    plt.plot(activity_df['SSE'])
    plt.title('SSE values across iterations')
    plt.xlabel('Iteration')
    plt.ylabel('SSE')
    plt.show()

    # Plotting No. of active centroids
    plt.plot(activity_df['No. of active centroids'])
    plt.title('No. of active centroids across iterations')
    plt.xlabel('Iteration')
    plt.ylabel('No. of active centroids')
    plt.show()

    # Plotting percent of active centroids
    plt.plot(activity_df['Percent of active centroids'])
    plt.title('Percent of active centroids across iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Percent of active centroids')
    plt.show()

    write_output(centroids, partition)

if __name__ == "__main__":
    main()
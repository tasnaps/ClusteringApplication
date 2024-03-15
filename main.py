import numpy as np
import requests as rq
import time
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

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
def random_centroids(data, k):
    """
    :param data: The data points to be clustered.
    :param k: The number of clusters to be created.
    :return: initial centroids.
    """
    # randomize indices without replacement for creating unique centroids
    random_idx = np.random.choice(data.shape[0], size=k, replace=False)
    centroids = data[random_idx]
    return centroids
def kmeans_plusplus(data, k):
    """
    :param data: The data points to be clustered.
    :param k: The number of clusters to be created.
    :return: initial centroids.
    """
    n_points = len(data)
    # Choose a center uniformly:
    centroids = [data[np.random.randint(n_points)]]
    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in data])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()

        # Choose a random data point as new center with weighted prob distribution
        # is based on distance from distance from existing centroids
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centroids.append(data[i])
    return np.array(centroids)

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
 KD tree (cKDTree) is built using the centroids of the clusters. The query() method from cKDTree is then used to find the nearest centroid for each provided data point. The method tree.query(data) returns a tuple where the first element is the distance to the closest centroid, and the second element is the index of the closest centroids in the original centroids array for each point in the data set. The function thus essentially performs a nearest neighbour search.
    :param data: numpy array of shape (N, D) representing the data points to be assigned to partitions.
    :param centroids: numpy array of shape (M, D) representing the centroids of the partitions.
    :return: numpy array of shape (N,) containing the index of the partition to which each data point is assigned.

    This method takes in a set of data points and a set of centroids and assigns each data point to the nearest centroid based on Euclidean distance. It uses a KDTree for efficient nearest
    * neighbor search.

    """
    tree = cKDTree(centroids)
    partition = tree.query(data)[1]
    return partition

def read_data_from_url(url):
    """
    Reads data from a URL and returns it.

    :return: The data retrieved from the URL.
    """
    try:
        file = rq.get(url)
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

def kmeans(data, k, init_method='kmeans++', use_random_swap=False):
    """
    Perform k-means clustering on the given data.

    :param data: The input data to cluster.
    :param k: The number of clusters to create.
    :return: A tuple containing the final centroids, partition, and activity tracker.


    """
    if init_method == 'kmeans++':
        centroids = kmeans_plusplus(data, k)
    elif init_method == 'random':
        centroids = random_centroids(data, k)
    elif init_method == 'grid':
        centroids = grid_based_centroids(data, k)
        if len(centroids) < k:
            print("Warning: Not enough grid centroids. Some clusters will be empty.")
        if len(centroids) > k:
            print("warning: more grid centroids than requested clusters. some centroids will be discarded.")
    else:
        raise ValueError("Invalid initialization method")

    partition = optimal_partition(data, centroids)
    activity_tracker = []
    if use_random_swap==False:
        while True:
            new_partition = optimal_partition(data, centroids)
            new_centroids = fetch_new_centroids(data, new_partition, k)

            # Calculate SSE and number of active centroids
            sse = sum_of_squared_errors(data, new_centroids, new_partition)
            active_centroids = np.count_nonzero(np.any(np.abs(centroids - new_centroids) > 0.001, axis=1))
            active_centroids_percent = (active_centroids / k) * 100

            # Track the values
            activity_tracker.append((sse, active_centroids, active_centroids_percent))

            # Check for convergence
            if np.all(new_centroids == centroids):
                break

            centroids = new_centroids
            partition = new_partition
    elif use_random_swap==True:
        old_sse = np.inf
        iterations_without_improvement = 0
        while True:
            centroids = random_swap(centroids, data)
            new_partition = optimal_partition(data, centroids)
            new_centroids = fetch_new_centroids(data, new_partition, k)

            sse = sum_of_squared_errors(data, new_centroids, new_partition)
            active_centroids = np.count_nonzero(np.any(np.abs(centroids - new_centroids) > 0.001, axis=1))
            active_centroids_percent = (active_centroids / k) * 100

            # Track the values
            activity_tracker.append((sse, active_centroids, active_centroids_percent))

            # Check for improvement in SSE
            if sse >= old_sse:
                iterations_without_improvement += 1
            else:
                iterations_without_improvement = 0
            old_sse = sse

            # If no improvement in last 4 iterations, break the loop
            if iterations_without_improvement >= 4:
                break

            # Update the centroids
            centroids = new_centroids


    return centroids, partition, activity_tracker

def grid_based_centroids(data, k):
    assert len(data) > 0

    # Create a grid of buckets
    numberOfSquaresForEachFeature = k
    grid = makeGrid(data, numberOfSquaresForEachFeature)
    bucket = putDataInSquares(data, grid)

    # Calculate the centroids for each grid square and collect them
    # Centroid of each grid square becomes the mean of all points in the square
    centroids = []
    for key, points in bucket.items():
        centroid = np.mean(np.vstack(points), axis=0)
        centroids.append(centroid)

    if len(centroids) > k:
        print("Selecting top-k centroids...")
        distances = scipy.spatial.distance.cdist(data, centroids, 'euclidean')
        nearest_centroids_ind = np.argmin(distances, axis=1)
        freq_count = np.bincount(nearest_centroids_ind)
        sorted_centroids = [x for _, x in sorted(zip(freq_count, range(len(freq_count))), reverse=True)]
        selected_centroids = sorted_centroids[:k]
        centroids = [centroids[i] for i in selected_centroids]
    return np.array(centroids)


# If we got more centroids than

def random_swap(centroids, data):
    old_centroids = centroids.copy()

    #Select random centroids
    centroid_idx = np.random.choice(len(centroids))

    #Select data point, not in centroids:
    new_centroids = data[np.random.choice(len(data))]

    #Check if new centroids are not used:
    while any(np.array_equal(new_centroids, old_centroids) for
              centroid in centroids):
                new_centroids = data[np.random.choice(len(data))]

    #Replace old centroid with new centroid
    centroids[centroid_idx] = new_centroids

    return centroids

def plotting(activity_tracker, use_random_swap_label, mode='ALL'):
    # Convert the activity tracker to a DataFrame for easy manipulation
    activity_df = pd.DataFrame(activity_tracker,
                               columns=['SSE', 'No. of active centroids', 'Percent of active centroids'])

    print(activity_df)

    # Plotting SSE values

    if mode=='SSE':
        plt.plot(activity_df['SSE'])
        plt.title('SSE values across iterations\n' + use_random_swap_label)
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        plt.show()
    else:
        plt.plot(activity_df['SSE'])
        plt.title('SSE values across iterations\n' + use_random_swap_label)
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        plt.show()
        # Plotting No. of active centroids
        plt.plot(activity_df['No. of active centroids'])
        plt.title('No. of active centroids across iterations\n' + use_random_swap_label)
        plt.xlabel('Iteration')
        plt.ylabel('No. of active centroids')
        plt.show()

        # Plotting percent of active centroids
        plt.plot(activity_df['Percent of active centroids'])
        plt.title('Percent of active centroids across iterations\n' + use_random_swap_label)
        plt.xlabel('Iteration')
        plt.ylabel('Percent of active centroids')
        plt.show()


def plot_sse_values(sse_values_data):
    plt.figure(figsize=(10, 5))
    color_map = cm.get_cmap('rainbow')
    colors = color_map(np.linspace(0, 1, len(sse_values_data[1])))
    for run in range(len(sse_values_data[1])):
        ks = []
        sses = []
        for k, sse_values in sorted(sse_values_data.items()):
            ks.append(k)
            sses.append(sse_values[run])
        plt.plot(ks, sses, marker='o', color=colors[run], label=f'Run {run + 1}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE Value')
    plt.title('Elbow Method showing the optimal k')
    plt.legend()
    plt.grid(True)
    plt.show()

#https://en.wikipedia.org/wiki/Levenshtein_distance
def edit_distance(string1, string2):
    #Create a two-dimensional list, all values 0
    distances = [[0 for _ in range(len(string2) + 1)] for _ in range(len(string1) + 1)]
    #Populate first row
    for i in range(len(string1) + 1):
        distances[i][0] = i
    #populate first column
    for j in range(len(string2) + 1):
        distances[0][j] = j
    #Iteration
    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            if string1[i - 1] == string2[j - 1]:
                cost = 0
            else:
                cost = 1
            distances[i][j] = min(distances[i - 1][j] + 1,  # Deletion
                                  distances[i][j - 1] + 1,  # Insertion
                                  distances[i - 1][j - 1] + cost)  # Substitution

    return distances[-1][-1]

def exercise4_4():
    A = ["ireadL", "relanE", "rlanZd", "irelLITnd"]
    B = ["fiInVlLand", "filanNM", "finPAlaQd", "finlCnUd"]

    print("Pairwise distances within Cluster A:")
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            distance = edit_distance(A[i], A[j])
            print(f"The edit distance between {A[i]} and {A[j]} is: {distance}")

    # Calculate all pairwise distances of items within Cluster B
    print("Pairwise distances within Cluster B:")
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            distance = edit_distance(B[i], B[j])
            print(f"The edit distance between {B[i]} and {B[j]} is: {distance}")

    medoid_A, cost_A = find_medoid_and_cost(A)
    medoid_B, cost_B = find_medoid_and_cost(B)
    print(f"Medoid for Cluster A is: {medoid_A} with cost: {cost_A}")
    print(f"Medoid for Cluster B is: {medoid_B} with cost: {cost_B}")

def find_medoid_and_cost(cluster):
    min_dist_sum = float('inf')
    medoid_str = None

    for i in range(len(cluster)):
        dist_sum = 0

        for j in range(len(cluster)):
            if i != j:
                dist_sum += edit_distance(cluster[i], cluster[j])

        if dist_sum < min_dist_sum:
            min_dist_sum = dist_sum
            medoid_str = cluster[i]

    return medoid_str, min_dist_sum

def exercise6_task1():
    data = read_data_from_url()
    # Exercise 6/7, 1. Apply k-means using different k-values. example: 1 to 10 k values, plot SSE, detect possible elbow/knee points.
    sse_values_data = {}
    runs = 20
    for run in range(runs):
        for k in range(1, 21):
            centroids, partition, activity_tracker = kmeans(data, k, use_random_swap=False)
            sse_values_data.setdefault(k, {})[run] = activity_tracker[-1][0]
    plot_sse_values(sse_values_data)

def exercise6_task2():
    data = read_data_from_url()
    sse_values = []
    for k in range(1, 21):
        centroids, partition, activity_tracker = kmeans(data, k, use_random_swap=False)
        sse_values.append(activity_tracker[-1][0])  # Append last SSE value from each iteration
    firs_derivative = np.diff(sse_values)
    second_derivative = np.diff(sse_values, n=2)
    print('First derivative:', firs_derivative)
    print('Second derivative:', second_derivative)

def exercise6_task3():
    data = read_data_from_url()
    sample_percentages = [1.0, 0.2, 0.1, 0.01]
    sse_values = {}
    for percent in sample_percentages:
        sample_size = int(len(data) * percent)
        data_sampled = data[np.random.choice(data.shape[0], sample_size, replace=False), :]
        centroids, partition, activity_tracker = kmeans(data_sampled, 16, use_random_swap=False)
        sse_values[percent] = activity_tracker[-1][0]
        plotting(activity_tracker, "Sample percentage: " + str(percent))
    return sse_values

def makeGrid(data, numberOfSquaresForEachFeature):
    grid = {}
    for attribute in range(data.shape[1]):
        smallestValue = np.min(data[:, attribute])
        biggestValue = np.max(data[:, attribute])
        sizeOfEachSquare = (biggestValue - smallestValue) / numberOfSquaresForEachFeature
        grid[attribute] = {
            "start": smallestValue,
            "end": biggestValue,
            "squareSize": sizeOfEachSquare
        }
    return grid


def putDataInSquares(data, grid):
    bucket = {}
    for i in range(data.shape[0]):
        piece = data[i]
        squareLocation = {}
        for attribute in range(data.shape[1]):
            squareNumber = int((piece[attribute] - grid[attribute]["start"]) / grid[attribute]["squareSize"])
            squareLocation[attribute] = squareNumber
        addPieceToSquare(piece, squareLocation, bucket)
    return bucket

def addPieceToSquare(piece, squareLocation, bucket):
    key = tuple(squareLocation.items()) # using the square location dict items as a key
    if key not in bucket:
        bucket[key] = []
    bucket[key].append(piece)

def divisiveAlgorithm(data, primary_k=1, secondary_k=2, sse_delta_threshold=0.01):
    primary_centroids, primary_partition, primary_tracker = kmeans(data, primary_k, use_random_swap=False)
    primary_sse = primary_tracker[-1][0]

    # now we will split each cluster until the sse reduction falls below the sse_delta_threshold
    current_clusters = [data]
    while True:
        new_clusters = []
        sse_reduction = 0
        for cluster in current_clusters:
            secondary_centroids, secondary_partition, secondary_tracker = kmeans(cluster, secondary_k,
                                                                                 use_random_swap=False)
            secondary_sse = secondary_tracker[-1][0]
            if secondary_sse / primary_sse > sse_delta_threshold:
                sse_reduction += primary_sse - secondary_sse
                new_clusters.extend([
                    cluster[secondary_partition == label]
                    for label in range(secondary_k)
                ])
            else:
                new_clusters.append(cluster)

        if sse_reduction < sse_delta_threshold:
            break

        current_clusters = new_clusters
    return current_clusters, secondary_tracker


def main():

    #exercise4_4()
    #k = int(input("Enter number of clusters: "))
    #centroids, partition, activity_tracker = kmeans(data, k, use_random_swap=True)
    #plotting(activity_tracker, "Random Swap Used")
    #write_output(centroids, partition)
    #centroids2, partition2, activity_tracker2 = kmeans(data, k, use_random_swap=False)
    #plotting(activity_tracker2, "Random swap not used")
    #exercise6_task3()
    data_s1 = read_data_from_url('https://cs.uef.fi/sipu/datasets/s1.txt')
    data_s2 = read_data_from_url('https://cs.uef.fi/sipu/datasets/s2.txt')
    data_s3 = read_data_from_url('https://cs.uef.fi/sipu/datasets/s3.txt')
    data_s4 = read_data_from_url('https://cs.uef.fi/sipu/datasets/s4.txt')
    #numberOfSquaresForEachFeature = 100
    #grid = makeGrid(data, numberOfSquaresForEachFeature)
    #bucket = putDataInSquares(data, grid)
    #centroids, partition, activity_tracker = kmeans(data, 5, init_method='random', use_random_swap=False)
    #plotting(activity_tracker, "Random initialization")
    #centroids2, partition2, activity_tracker2 = kmeans(data, 5, init_method='grid', use_random_swap=False)
    #plotting(activity_tracker2, "Grid initialization")
    ##Random initialization for k-means

    # S1
    start_time_s1 = time.time()
    result_s1, tracker_s1 = divisiveAlgorithm(data_s1)
    end_time_s1 = time.time()
    process_time_s1 = end_time_s1 - start_time_s1

    # S2
    start_time_s2 = time.time()
    result_s2, tracker_s2 = divisiveAlgorithm(data_s2)
    end_time_s2 = time.time()
    process_time_s2 = end_time_s2 - start_time_s2

    # S3
    start_time_s3 = time.time()
    result_s3, tracker_s3 = divisiveAlgorithm(data_s3)
    end_time_s3 = time.time()
    process_time_s3 = end_time_s3 - start_time_s3

    # S4
    start_time_s4 = time.time()
    result_s4, tracker_s4 = divisiveAlgorithm(data_s4)
    end_time_s4 = time.time()
    process_time_s4 = end_time_s4 - start_time_s4
    print(f"Process time for S1: {process_time_s1}")
    print(f"Process time for S2: {process_time_s2}")
    print(f"Process time for S3: {process_time_s3}")
    print(f"Process time for S4: {process_time_s4}")


    plotting(tracker_s1, "divisiveAlgorithm_s1", mode='SSE')
    plotting(tracker_s2, "divisiveAlgorithm_s2", mode='SSE')
    plotting(tracker_s3, "divisiveAlgorithm_s3", mode='SSE')
    plotting(tracker_s4, "divisiveAlgorithm_s4", mode='SSE')




if __name__ == "__main__":
    main()
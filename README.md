# KMeans Clustering Python

This repository contains a Python implementation of the K-Means clustering algorithm. It is intended to work with data fetched from a URL and assigns each data point to its nearest centroid based on Euclidean distance. There's also functionality to visualize the clustering process in the form of line graphs, showing SSE values, the number of active centroids, and the percentage of active centroids throughout the iterations.

## Project Files

* `kmeans.py`: The main Python script containing implementation of K-means clustering algorithm along with necessary helper functions.

## Key Functions

- `parse_data`: Parses the given text (expected to be data points, separated by newline) and returns them as a numpy array.
- `distance_function`: Calculates the Euclidean distance between two numpy arrays, useful for determining which cluster a data point belongs to.
- `sum_of_squared_errors`: Calculate the sum of squared errors for a given dataset and clustering.
- `kmeans_plusplus`: A function to generate an initial clustering of the data for initial step of K-means.
- `write_output`: Write the centroids and partition data to files.
- `optimal_partition`: Assigns each data point to the nearest centroid based on Euclidean distance.
- `read_data_from_url`: Reads data from a URL and returns it as a numpy array after parsing.
- `fetch_new_centroids`: Updates the centroid positions based on the current partition of the data.
- `kmeans`: Performs the K-means clustering on the given data.
- `main`: Asks user for number of clusters, applies K-means to data, plots progress of algorithm, and finally writes the output to files.

## Usage

Clone the repository to your local machine and navigate to its directory. Then, run the `main.py` file

Note: You will be prompted to enter the number of clusters when you run the script.

## Requirements 

The implementation uses the following Python libraries:
1. numpy
2. requests
3. scipy
4. pandas
5. matplotlib

## Limitations

Currently, the URL of the dataset is hard-coded into the `read_data_from_url` function, and implementation works only for this specific data format. For a different dataset, you may need to modify this function accordingly.

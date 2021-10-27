import math
import random

import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(sample1, sample2):
    distance = 0
    for i in range(len(sample1)):
        distance += (sample1[i] - sample2[i]) ** 2
    return math.sqrt(distance)


def find_centroids(dataset, clusters):
    classes = {}
    centroids = []
    for i in range(len(clusters)):
        if clusters[i] in classes.keys():
            classes[clusters[i]].append(dataset[i])
        else:
            classes[clusters[i]] = [dataset[i]]
    print("CLASSES", classes)
    for class_label in classes.keys():
        points = classes[class_label]
        print("POINTS", points)
        centroid = np.mean(points, axis=0)
        print("CENTROID", centroid)
        centroids.append(centroid)

    return centroids

def check_equal(arr1, arr2):
        eps = 0.000001
        for i in range(len(arr1)):
            for j in range(len(arr1[i])):
                if math.abs(arr1[i][j] - arr2[i][j]) > eps:
                    return False
        return True



def k_means_clustering(dataset, k):
    m = dataset.shape[0]
    n = dataset.shape[1]
    centroids = dataset[random.sample(range(m), k,)]
    n_iter = 10
    clusters = []
    for iter in range(n_iter):
        print(centroids)
        prev_centroids = centroids.copy()
        for idx, point in enumerate(dataset):
            distances = []

            if check_equal(centroids, prev_centroids):
                break

            
            for centroid_idx, centroid in enumerate(centroids):
                distances.append((euclidean_distance(point, centroid), centroid_idx))
            sorted_distances = sorted(distances)
            nearest_centroid = sorted_distances[0]
            nearest_centroid_cluster = nearest_centroid[1]
            clusters.append(nearest_centroid_cluster)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=clusters, cmap='rainbow')
            plt.show()

        centroids = find_centroids(dataset, clusters)
        print("CLUSTERS", clusters)
        

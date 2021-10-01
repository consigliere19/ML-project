import math
import numpy as np


def euclidean_distance(sample1, sample2):
    distance = 0
    for i in range(len(sample1)):
        distance += (sample1[i] - sample2[i]) ** 2
    return math.sqrt(distance)


def find_centroids(X_train, y_train):
    classes = {}
    centroids = []
    for i in range(len(y_train)):
        classes[y_train[i]].append(X_train[i])
    for class_label in classes.keys():
        points = classes[class_label]
        centroid = np.mean(points, axis=0)
        centroids.append((centroid, class_label))
    return centroids


def predict(X, centroids):
    distances = []
    min_dist = float('inf')
    for centroid, class_label in centroids:
        dist = euclidean_distance(X, centroid)
        if dist < min_dist:
            min_dist = dist
            pred_class = class_label
    return pred_class


def mdc_classifier(X_train, y_train, X_test):
    centroids = find_centroids(X_train, y_train)
    predictions = []
    for X in X_test:
        prediction = predict(X, centroids)
        predictions.append(prediction)
    return predictions

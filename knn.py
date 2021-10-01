import math



def euclidean_distance(sample1, sample2):
    distance = 0
    for i in range(len(sample1)):
        distance += (sample1[i] - sample2[i]) ** 2
    return math.sqrt(distance)


def predict(X_train, y_train, query, n_neighbors):
    distances = []
    for idx, sample in enumerate(X_train):
        distance = euclidean_distance(sample, query)
        distances.append((distance, idx))
    sorted_distances = sorted(distances)
    k_nearest = sorted_distances[:n_neighbors]
    k_nearest_labels = [y_train[i] for distance, i in k_nearest]
    most_freq = 0
    for temp in k_nearest_labels:
        if temp > most_freq:
            most_freq = temp


def knn_classifier(X_train, y_train, X_test, n_neighbors):
    predictions = []
    for X in X_test:
        prediction = predict(X_train, y_train, X, n_neighbors)
        predictions.append(prediction)
    return predictions

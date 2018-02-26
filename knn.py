import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
from sklearn import neighbors

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]

# Graph percentage correct vs training set size
# x = []
# y = []
# sizes = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
# for size in sizes:
#     train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=size)

#     train_images[train_images > 0] = 1
#     test_images[test_images > 0] = 1

#     knn = neighbors.KNeighborsClassifier(p=1)
#     print("loading data")
#     knn.fit(train_images, train_labels.values.ravel())

#     score = knn.score(test_images, test_labels.values.ravel())
#     x.append(size)
#     y.append(score)

# plt.plot(x, y)
# plt.ylabel("Percentage correct in test set")
# plt.xlabel("Training set size")
# plt.show()

# Using training set size of 1000, graph percentage correct vs number of neighbors
# x = []
# y = []
# numbers_of_neighbors = range(1, 11)
# for num_neighbors in numbers_of_neighbors:
#     scores = []

#     for _ in range(3):
#         train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=1000)

#         train_images[train_images > 0] = 1
#         test_images[test_images > 0] = 1

#         knn = neighbors.KNeighborsClassifier(p=1, n_neighbors=num_neighbors)
#         print("loading data")
#         knn.fit(train_images, train_labels.values.ravel())

#         score = knn.score(test_images, test_labels.values.ravel())
#         scores.append(score)

#     average_score = sum(scores) / len(scores)
#     x.append(num_neighbors)
#     y.append(average_score)

# plt.plot(x, y)
# plt.ylabel("Percentage correct in test set")
# plt.xlabel("Number of Neighbors")
# plt.show()

# Binary cutoff point for image pixels vs percentage correct
x = []
y = []
cutoff_points = [x / 10.0 for x in range(0, 10)]
for cutoff_point in cutoff_points:
    scores = []

    for _ in range(3):
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=1000)

        train_images[train_images > cutoff_point] = 1
        test_images[test_images > cutoff_point] = 1
        train_images[train_images <= cutoff_point] = 0
        test_images[test_images <= cutoff_point] = 0

        knn = neighbors.KNeighborsClassifier(p=1, n_neighbors=3)
        print("loading data")
        knn.fit(train_images, train_labels.values.ravel())

        score = knn.score(test_images, test_labels.values.ravel())
        scores.append(score)

    average_score = sum(scores) / len(scores)
    x.append(cutoff_point)
    y.append(average_score)

plt.plot(x, y)
plt.ylabel("Percentage correct in test set")
plt.xlabel("Cutoff Point")
plt.show()

# Vary number of neighbors
# Vary cutoff/granularity
# Vary amount of data we "train" on
# Training time
# 1.4333410263061523
# scoring
# 0.6
# Scoring time
# 8.369163751602173

# For next week:
# Look at/produce visual errors (misclassifications)
# Decision trees/neural networks?
# KNN/neural networks
# SVM

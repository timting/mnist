import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
from sklearn import neighbors

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]

# Using training set size of 1000, graph percentage correct vs number of neighbors
x = []
y = []
numbers_of_neighbors = range(1, 11)
for num_neighbors in numbers_of_neighbors:
    scores = []

    for _ in range(3):
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=1000)

        train_images[train_images > 0] = 1
        test_images[test_images > 0] = 1

        knn = neighbors.KNeighborsClassifier(p=1, n_neighbors=num_neighbors)
        print("loading data")
        knn.fit(train_images, train_labels.values.ravel())

        score = knn.score(test_images, test_labels.values.ravel())
        scores.append(score)

    average_score = sum(scores) / len(scores)
    x.append(num_neighbors)
    y.append(average_score)

plt.plot(x, y)
plt.ylabel("Percentage correct in test set")
plt.xlabel("Number of Neighbors")
plt.show()
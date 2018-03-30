import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
from sklearn import neighbors

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]

# Binary cutoff point for image pixels vs percentage correct
x = []
y = []
cutoff_points = [x / 20.0 for x in range(0, 20)]
for cutoff_point in cutoff_points:
    scores = []

    for _ in range(5):
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

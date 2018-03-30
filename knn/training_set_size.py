import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
from sklearn import neighbors

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]

# Graph percentage correct vs training set size
x = []
y = []
sizes = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
for size in sizes:
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=size)

    train_images[train_images > 0] = 1
    test_images[test_images > 0] = 1

    knn = neighbors.KNeighborsClassifier(p=1)
    print("loading data")
    knn.fit(train_images, train_labels.values.ravel())

    score = knn.score(test_images, test_labels.values.ravel())
    x.append(size)
    y.append(score)

plt.plot(x, y)
plt.ylabel("Percentage correct in test set")
plt.xlabel("Training set size")
plt.show()
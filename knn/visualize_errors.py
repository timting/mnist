import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
from sklearn import neighbors

# Visualize errors
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=1000)

train_images[train_images > 0.3] = 1
test_images[test_images > 0.3] = 1
train_images[train_images <= 0.3] = 0
test_images[test_images <= 0.3] = 0

knn = neighbors.KNeighborsClassifier(p=1, n_neighbors=3)
print("loading data")
knn.fit(train_images, train_labels.values.ravel())

actuals = test_labels.values.ravel()
predictions = knn.predict(test_images)
for idx, prediction in enumerate(predictions):
    if prediction != actuals[idx]:
        fig=plt.figure(figsize=(8, 8))
        test_img = test_images.iloc[idx].as_matrix()
        test_img = test_img.reshape((28,28))
        fig.add_subplot(2, 3, 2)
        plt.imshow(test_img,cmap='gray')
        plt.title("pred: %s, actual: %s" % (prediction, actuals[idx]))
        nn_indices = knn.kneighbors([test_images.iloc[idx]], 3, False)
        print(idx)
        print(nn_indices)
        for j, nn_index in enumerate(nn_indices[0]):
            train_img = train_images.iloc[nn_index].as_matrix()
            train_img = train_img.reshape((28,28))
            fig.add_subplot(2, 3, j + 4)
            plt.imshow(train_img,cmap='gray')
            plt.title(train_labels.iloc[nn_index,0])
        plt.show()
        input()
        #print("test[%d] indices: %s" % (idx, ", ".join(nn_indices[0])))

import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage import exposure
import imutils
import cv2

# KILDER:
# https://martin-thoma.com/svm-with-sklearn/
# https://scikit-learn.org/stable/modules/svm.html

mnist = datasets.load_digits()

x = mnist.data
y = mnist.target

# Scale data to [-1, 1] - This is of mayor importance!!!
x = x / 255.0 * 2 - 1

x, y = shuffle(x, y, random_state=0)


(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                mnist.target, test_size=0.25, random_state=42)


clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=.0073)

print("Start fitting. This may take a while")

# take all of it - make that number lower for experiments

clf.fit(trainData, trainLabels)

predicted = clf.predict(testData)
print("Confusion matrix:\n", confusion_matrix(testLabels.ravel(), predicted))

print('\n', classification_report(testLabels.ravel(), predicted))
print("\nAccuracy: ",  clf.score(testData, testLabels))

for i in np.random.randint(0, high=len(testLabels), size=(5,)):
    image = testData[i]
    prediction = clf.predict(image.reshape(1, -1))[0]

    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from skimage import exposure
import imutils
import cv2

mnist = datasets.load_digits()


(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                mnist.target, test_size=0.25, random_state=42)

clf = MultinomialNB()
clf.fit(trainData, trainLabels)

y_pred = clf.predict(testData)
y_true = testLabels.ravel()

print('Confusion Matrix: \n', confusion_matrix(y_true, y_pred))
print('\nClassification Report:\n', classification_report(y_true, y_pred))
print('MultinomailNB accuracy: ', clf.score(testData, testLabels))

for i in np.random.randint(0, high=len(testLabels), size=(5,)):
    image = testData[i]
    prediction = clf.predict(image.reshape(1, -1))[0]

    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
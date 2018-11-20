from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2

mnist = datasets.load_digits()
# print(mnist.data)

# 75% for trening og 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                mnist.target, test_size=0.25, random_state=42)

# Tar 10% av trenings dataen og brukes til validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# Ser på størrelsen for hver split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

#Initialiserer k får vår knn classifier
kVals = range(1, 30, 2)
accuracies = []

# Looper over kVals
for k in range(1, 30, 2):
    # Tren the classifier med valuen til "k"
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    #Evaluate moddelen og print
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# Største accuracy
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

# Nå som vi har den beste K verdien, tren classifieren på nytt
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)

# Predict labels for test settet
predictions = model.predict(testData)

# Evaluate performance of model for each of the digits
print("\nEVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

# some indices are classified correctly 100% of the time (precision = 1)
# high accuracy (98%)

# check predictions against images
# loop over a few random digits
for i in np.random.randint(0, high=len(testLabels), size=(5,)):
    # np.random.randint(low, high=None, size=None, dtype='l')
    image = testData[i]
    prediction = model.predict(image.reshape(1, -1))[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels for better visualization
    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0) # press enter to view each one!
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import imutils
import cv2
from skimage import exposure


mnist = fetch_mldata('MNIST original')


(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                mnist.target, test_size=0.25, random_state=42)

clf = MultinomialNB()
clf.fit(trainData, trainLabels)

print(trainData.shape)
print(testData.shape)

y_pred = clf.predict(testData)
y_true = testLabels.ravel()

print(classification_report(y_true, y_pred))
print('\nMultinomial Naive bayes accuracy: ', accuracy_score(y_true, y_pred))

'''
for i in np.random.randint(0, high=len(testLabels), size=(10,)):
    # np.random.randint(low, high=None, size=None, dtype='l')
    image = testData[i]
    prediction = clf.predict(image.reshape(1, -1))[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels for better visualization
    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)  # press enter to view each one!
'''
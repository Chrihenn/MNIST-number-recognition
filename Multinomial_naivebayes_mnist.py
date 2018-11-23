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
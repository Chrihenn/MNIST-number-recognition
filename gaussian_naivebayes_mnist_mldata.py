from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import numpy as np

mnist = fetch_mldata('MNIST original')
# print(mnist.data)


(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                mnist.target, test_size=0.25, random_state=42)

clf = GaussianNB()
clf.fit(trainData, trainLabels)

y_pred = clf.predict(testData)
y_true = testLabels.ravel()


print('Confusion Matrix: \n', confusion_matrix(y_true, y_pred))
print('\nClassification Report:\n', classification_report(y_true, y_pred))
print('GaussianNB accuracy: ', clf.score(testData, testLabels))


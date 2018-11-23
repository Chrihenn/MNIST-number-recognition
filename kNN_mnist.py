from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import fetch_mldata
import numpy as np


mnist = fetch_mldata('MNIST original')
# print(mnist.data)

# 75% for trening og 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),mnist.target, test_size=0.25,
                                                                  random_state=42)

# Tar 10% av trenings dataen og brukes til validation

(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# Ser på størrelsen for hver split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# Initialiserer k får vår knn classifier
kVals = range(1, 30, 2)
accuracies = []

# Looper over kVals
for k in range(1, 30, 2):
    # Tren the classifier med valuen til "k"
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # Evaluate moddelen og print
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# Største accuracy
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

# Nå som vi har den beste K verdien, tren classifieren på nytt
model = KNeighborsClassifier(n_neighbors=kVals[i], algorithm='brute')
model.fit(trainData, trainLabels)

# Predict labels for test settet
predictions = model.predict(testData)

print('Confusion Matrix: \n', confusion_matrix(testLabels.ravel(), predictions))
print('\nClassification Report:\n', classification_report(testLabels.ravel(), predictions))
print('kNN accuracy: ', model.score(testData, testLabels))


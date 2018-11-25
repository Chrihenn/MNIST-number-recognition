import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

# KILDER:
# https://martin-thoma.com/svm-with-sklearn/
# https://scikit-learn.org/stable/modules/svm.html

mnist = fetch_mldata('MNIST original')

x = mnist.data
y = mnist.target

# Scale data to [-1, 1] - This is of mayor importance!!!
x = x / 255.0 * 2 - 1

x, y = shuffle(x, y, random_state=0)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
data = {'train': {'X': x_train,
                  'y': y_train},
        'test': {'X': x_test,
                 'y': y_test}}
# Get classifier

clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=.0073)

print("Start fitting. This may take a while")

# take all of it - make that number lower for experiments
examples = len(data['train']['X'])
clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])

predicted = clf.predict(data['test']['X'])
print("Confusion matrix:\n%s" %
metrics.confusion_matrix(data['test']['y'], predicted))

print('\n', classification_report(data['test']['y'], predicted))
print("\nAccuracy: %0.4f" % metrics.accuracy_score(data['test']['y'], predicted))
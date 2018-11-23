from __future__ import division
import numpy as np
from sklearn import tree
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals.six import io
from sklearn.tree import export_graphviz
import pydotplus


mndata = fetch_mldata('MNIST original')

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mndata.data),
                                                mndata.target, test_size=0.25, random_state=42)

clf_dt = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
clf_dt.fit(trainData, trainLabels)

pred = clf_dt.predict(testData)

print('Confusion Matrix: \n', confusion_matrix(testLabels.ravel(), pred))
print('\nClassification Report:\n', classification_report(testLabels.ravel(), pred))
print('DTREE accuracy: ', clf_dt.score(testData, testLabels))

'''
dot_data = io.StringIO()

export_graphviz(clf_dt, out_file=dot_data, filled=True, rounded=True, special_characters=True)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png("dtree.png")
'''

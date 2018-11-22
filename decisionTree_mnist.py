from __future__ import division
import numpy as np
import pandas as pd
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.externals.six import io
from sklearn.tree import export_graphviz
import pydotplus
from PIL import Image


mndata = datasets.load_digits()

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mndata.data),
                                                mndata.target, test_size=0.25, random_state=42)

clf_dt = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
clf_dt.fit(trainData, trainLabels)

print(clf_dt.score(testData, testLabels))

dot_data = io.StringIO()

export_graphviz(clf_dt, out_file=dot_data, filled=True, rounded=True, special_characters=True)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png("dtree.png")


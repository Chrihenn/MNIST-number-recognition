from __future__ import division
import numpy as np
from sklearn import tree
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.externals.six import io
from sklearn.tree import export_graphviz
import pydotplus
from skimage import exposure
import imutils
import cv2

mndata = fetch_mldata('MNIST original')

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mndata.data),
                                                mndata.target, test_size=0.25, random_state=42)

clf_dt = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
clf_dt.fit(trainData, trainLabels)

print('DTREE accuracy: ', clf_dt.score(testData, testLabels))
'''
dot_data = io.StringIO()

export_graphviz(clf_dt, out_file=dot_data, filled=True, rounded=True, special_characters=True)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png("dtree.png")
'''


for i in np.random.randint(0, high=len(testLabels), size=(10,)):
    # np.random.randint(low, high=None, size=None, dtype='l')
    image = testData[i]
    prediction = clf_dt.predict(image.reshape(1, -1))[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels for better visualization
    image = image.reshape((392, -1)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, height=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)  # press enter to view each one!


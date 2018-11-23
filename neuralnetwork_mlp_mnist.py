import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
import imutils
import cv2
from skimage import exposure
import os.path

PATH = 'mlp_model.pkl'


if __name__ == '__main__':
    print('Fetching and loading MNIST data')
    mnist = fetch_mldata('MNIST original')
    # mnist = datasets.load_digits()
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                      mnist.target, test_size=0.25, random_state=42)

    print('Got MNIST with %d training- and %d test samples' % (len(y_train), len(y_test)))
    print('Digit distribution in whole dataset:', np.bincount(y.astype('int64')))

    clf = None
    if os.path.exists(PATH):
        print('Loading model from file.')
        clf = joblib.load(PATH).best_estimator_
    else:
        print('Training model.')
        params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
        mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
        clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)
        print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
        print('Best params appeared to be', clf.best_params_)
        joblib.dump(clf, PATH)
        clf = clf.best_estimator_

    pred = clf.score(X_test, y_test)
    print('Test accuracy:', pred)

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
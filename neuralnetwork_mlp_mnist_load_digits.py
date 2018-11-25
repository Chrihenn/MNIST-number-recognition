import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage import exposure
import imutils
import cv2

# PATH = 'mlp_model.pkl'

if __name__ == '__main__':
    print('Fetching and loading MNIST data')
    mnist = datasets.load_digits()
    # mnist = datasets.load_digits()
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                      mnist.target, test_size=0.25, random_state=42)

    print('Got MNIST with %d training- and %d test samples' % (len(y_train), len(y_test)))
    print('Digit distribution in whole dataset:', np.bincount(y.astype('int64')))

    clf = None

    print('Training model.')
    params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
    mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
    clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
    clf.fit(trainData, trainLabels)
    print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
    print('Best params appeared to be', clf.best_params_)
    clf = clf.best_estimator_


    y_true = testLabels.ravel()
    y_pred = clf.predict(testData)

    print('Confusion Matrix: \n', confusion_matrix(y_true, y_pred))
    print('\nClassification Report:\n', classification_report(y_true, y_pred))
    print('MLP accuracy: ', clf.score(testData, testLabels))

    for i in np.random.randint(0, high=len(testLabels), size=(5,)):
        image = testData[i]
        prediction = clf.predict(image.reshape(1, -1))[0]

        image = image.reshape((8, 8)).astype("uint8")
        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

        print("I think that digit is: {}".format(prediction))
        cv2.imshow("Image", image)
        cv2.waitKey(0)

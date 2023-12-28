import numpy as np



def _weighted_maximum(x, w, m):

    return np.bincount(x, weights = w, minlength = m)


def update_weights_(w, alpha, y, y_pred):

    a_hx = np.where(y == y_pred, -alpha, alpha)
    w *= np.exp(a_hx)
    w /= np.sum(w)

    return w


class K_neighbours_classifier:

    def __init__(self, k = 10, alpha = None):

        self.k = k
        self.alpha = alpha


    def fit(self, X, y, w=None):

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.X = X
        self.y = y
        self.w = w

        return self


    def predict(self, X_test, y_test=None):

        y_pred = np.empty(X_test.shape[0])

        if self.w is None:

            for i, t in enumerate(X_test):
                k_nearest_distances = np.argsort([np.linalg.norm(x-t) for x in self.X])[:self.k]
                k_nearest_neighbours = np.array([self.y[i] for i in k_nearest_distances]).astype('int')
                y_pred[i] = np.bincount(k_nearest_neighbours, minlength = self.n_classes_).argmax()

        else:

            for i, t in enumerate(X_test):
                k_nearest_distances = np.argsort([np.linalg.norm(x-t) for x in self.X] * self.w)[:self.k]
                k_nearest_neighbours = np.array([self.y[i] for i in k_nearest_distances]).astype('int')
                y_pred[i] = np.bincount(k_nearest_neighbours, minlength = self.n_classes_).argmax()

        return y_pred
   
    
class Adaboost:

    def __init__(self, n_estimators_ = None):

        self.n_estimators_ = n_estimators_

    def fit(self, X, y):

        n_samples_, n_features_ = X.shape[0], X.shape[1]
        self.n_classes = np.unique(y).shape[0]
        if self.n_estimators_ is None:
            self.n_estimators_ = int(np.sqrt(n_samples_))
        w = np.ones(n_samples_) / n_samples_
        self.clfs = []

        for k in range(1, self.n_estimators_):

            clf = K_neighbours_classifier (k=k)
            clf.fit(X, y, w)
            pred = clf.predict(X)
            clf_error = (pred != y).sum() / n_samples_
            clf.alpha = 0.5 * np.log((1.0 - clf_error) / (clf_error + 1e-10))
            w = update_weights_(w, clf.alpha, y, pred)
            self.clfs.append(clf)

        return self


    def predict(self, X, y=None):

        n_samples = X.shape[0]
        y_pred = np.empty((self.n_estimators_ -1, n_samples))
        clf_w = np.empty(self.n_estimators_ - 1)
        for i, clf in enumerate (self.clfs):
            y_pred[i] = clf.predict(X)
            clf_w[i] = clf.alpha
        y_pred = y_pred.astype('int')
        predictions = np.array([(_weighted_maximum(x, clf_w, self.n_classes).argmax()) for x in y_pred.T])

        return predictions
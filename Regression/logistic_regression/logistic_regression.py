import numpy as np


class LogisticRegression:
    def __init__(self, ld=0, stepsize=3, iters=1e4, tol=1):
        """
        Constructor for LogisticRegression class
        :param ld: (float) regularization weighting
        :param stepsize: (float) Gradient descent step size
        :param iters: (int) Maximum iterations
        :param tol: (float) Convergence criteria
        """
        self.stepsize = stepsize
        self.ld = ld
        self.tol = tol
        self.iters = np.int(iters)
        self.B = None

    def predict(self, X, threshold=0.5):
        """
        Binary predictions
        :param X: (numpy array) input vectors
        :param threshold: (float) classification threshold
        :return: (vector) Vector of binary predictions
        """
        return np.digitize(expit(X @ self.B.T), [threshold], True)

    def predict_proba(self, X):
        """
        Continuous/probabilistic redictions
        :param X: (array) input vectors
        :return: (vector) Vector of estimated probabilities
        """
        return expit(X @ self.B.T)

    def GD_step(self, X, B, y):
        """
        A single iteration of gradient descent
        :param X: (array) input vectors
        :param B: (vector) LR coefficients
        :param y: (vector) targets
        """
        n = len(y)
        h = expit(X @ B)

        # Cost function gradient
        deltaJ = (self.stepsize / n) * (X.T @ (h - y)) + 2 * self.ld * B
        return (B - self.stepsize * deltaJ)

    def NLL(self, X, B, y):
        """
        Calculate the cost as total negative log-liklihood
        :param X: (array) input vectors
        :param B: (vector) LR coefficients
        :param y: (vector) targets
        """
        type1 = sum(-y * np.log(expit(X @ B)))
        type2 = sum((1 - y) * np.log(1 - expit(X @ B)))
        L2 = sum(self.ld * B ** 2)
        cost = np.nansum([type1.T, -type2.T]) + L2
        return cost

    def accuracy_breakdown(self, X, y, threshold=0.5):
        """
        This returns a summary of the confusion matrix
        :parm X: (array) input vectors
        :param y: (vector) targets
        :returns: (dict) breakdown of true and false positive and negatives
        """
        result, count = np.unique(y - self.predict(X, threshold=threshold) * 0.1, return_counts=True)
        return dict(zip(['False Positive', 'True Negative', 'True Positive', 'False Negative'], count))

    def fit(self, X, y, verbose=True):
        """
        Perform a GD fit
        :param X: (array) input vectors
        :param y: (vector) targets
        :param B: (vector) LR coefficients:
        :param verbose: (bool) verbosity
        """
        conv = np.zeros(self.iters)
        i = 0
        if self.B == None:
            self.B = np.zeros(len(X.T)).T

        while True:
            self.B = self.GD_step(X, self.B, y)
            conv[i] = self.NLL(X, self.B, y)
            if ((conv[i] - conv[i - 1]) > conv[i] * 1.44):
                print("Diverging!")
                return conv
            if (np.abs(conv[i - 1] - conv[i]) < self.tol):
                if verbose:
                    print("Converged in %s iterations!" % i)
                conv = conv[0:i]
                return conv
            if i == self.iters - 1:
                print("Max iters reached! ")
                conv = conv[0:i]
                return conv
            if verbose:
                print("\rIter: %s, Cost: %.2f" % (i, conv[i]), end=" ")
            i += 1

    def cross_validate(self, X, y, folds=5, verbose=True, **kwargs):
        """
        Crossvalidate the model
        :param X: (matrix) Input vectors
        :param y: (vector) Targets
        :param folds: (int) Number of CV iterations
        :param verbose: (bool) Verbosity
        :param **kwargs: (dict) Keywords for self.fit
        :return: (float) Estimated tests accuracy
        """
        # TODO impliment arbitrary metric function
        results = np.zeros(folds)
        chunks = chunk(np.hstack((X, np.array([y]).T)), folds)
        for i in range(len(chunks)):
            val_model = LogisticRegression(self.ld, self.stepsize, self.iters, self.tol)
            X_train = np.vstack(chunks[np.arange(folds) != i])[:, 0:-1]
            y_train = np.vstack(chunks[np.arange(folds) != i])[:, -1]

            X_test = chunks[i][:, 0:-1]
            y_test = chunks[i][:, -1]

            val_model.fit(X_train, y_train, verbose=False, **kwargs)
            results[i] = val_model.accuracy(X_test, y_test)

            if verbose:
                print("\rIter %s accuracy: %.3f" % (i, results[i]))
        mean = np.mean(results)
        std = np.std(results)

        if std > 0.05:
            print("Warning - low confidence")
        return np.mean(results)

    def accuracy(self, X, y, threshold=0.5):
        """
        Calculates accuracy
        :param X: (array) Input vectors
        :param y: (vector) Labels
        :return: (float) accuracy
        """
        matrix = y == self.predict(X, threshold=threshold)
        return np.mean(matrix)

    def print_confusion_matrix(self, X, y, threshold=0.5):
        """
        Prints confusion matrix
        :param X: (array) Input vectors
        :param y: (vector) targets
        """
        breakdown = self.accuracy_breakdown(X, y, threshold=threshold)
        print("~~~~~~~~~~~~~~")
        print('{{{:6}'.format('TP') + '{:>6}}}'.format('FP'))
        print('{{{:6}'.format(breakdown['True Positive']) + '{:>6}}}'.format(breakdown['False Positive']))
        print('{{{:6}'.format(breakdown['False Negative']) + '{:>6}}}'.format(breakdown['True Negative']))
        print('{{{:6}'.format('FN') + '{:>6}}}'.format('TN'))
        print("~~~~~~~~~~~~~~")


### Other tools

def expit(X, epsilon=1e-15):
    """
    Expit function with slight nudge to prevent numerical errors
    :param X: (array) Input vectors
    :param epsilon: (float) Nudget coefficient
    """
    from scipy.special import expit as spexpit
    return spexpit(X) - (spexpit(X) - 0.5) * epsilon


def chunk(X, chunks):
    """
    Shuffles an array into chunks for cross-validation
    :param X: input vectors
    :param chunks: number of fold
    """
    np.random.shuffle(X)
    return np.asarray(np.array_split(X, chunks, 0))

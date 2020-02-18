import numpy as np


class LinearRegression:
    """
    This class performs OLS linear regression
    """

    def __init__(self):
        """
        Constructor
        Sets self.trained to zero
        """
        self.trained = False
        self.B = None
    
    def rss(self, X, y):
        """
        Calculate residual sum of squares as cost function
        :param X: (array) Feature matrix
        :param y: (vector) targets
        """
        value = (y - (self.B @ X.T))   
        value = np.multiply(value, value)
        return np.sum(value)

    def predict(self, X):
        """
        Estimate values! 
        :param X: (array) Feature matrix
        """
        try: 
            y_est = self.B @ X.T
            return y_est
        except ValueError: print("Error - model has not been trained")

    def r_squared(self, X, y):
        """
        Calculates the coefficient of determination
        :param X: (array) Feature matrix
        :param y: (vector) Targets
        :return: (float) coefficient of determination
        """
        return 1 - self.rss(X, y) / np.sum((y - np.mean(y))**2)

    def grss(self, X, y):
        """
        Calculate the gradient of the cost function (analytically)
        :param X: (array) Feature matrix
        :param y: (vector) Targets
        """
        
        # d/dB (y-BX)**2 = 2 (y-BX)X
        return 2 * X.T @ (y - (self.B @ X.T)).T
        
    def fit(self, X, y, max_iters=10000, tol=1e-3, learning_rate=1e-5):
        """
        Fits a regression of y onto X
        :param X: (array) Feature matrix
        :param y: (vector) targets 
        :param max_iters: (int) Maximum iterations, default 1E5
        :param tol: (float) Tolerance for convergence
        :param learning_rate: (float) Learning rate, default 1E-5
        """
        
        # Initialize coefficients, cost, and iterations 
        self.B = np.zeros((np.shape(X)[1]))
        cost = self.rss(X, y)
        convergence = np.zeros(max_iters+1)
        convergence[0] = cost
        i = 0
        
        for i in range(max_iters):
            previous_cost = cost
            dRSS = self.grss(X, y)
            self.B = self.B + dRSS / (np.mean(X, axis=0)) * learning_rate
            cost = self.rss(X, y)
            convergence[i] = cost

            print("Cost: %.2f" % cost, end='\r')

            if abs(previous_cost - cost) < tol:
                self.trained = True
                return self.B,convergence[0:i]

            if cost > previous_cost * 200:
                print("\n Diverging!")
                return self.B,convergence[0:i]
        
        # If we reach here, the regression has neither converged nor diverged in the maximum number of iterations
        print("\n Max Iters Reached")
        return self.B,convergence[0:i]
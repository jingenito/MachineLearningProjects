import numpy as np 

from Metrics import accuracy_score

class LinearRegression:
    """A class that represents a multi-linear regression model."""

    def __init__(self):
        #initialize
        self.coeff = []

    def get_params(self) -> np.array:
        """Returns the parameters of the model."""
        return self.coeff

    def fit(self, X : np.array, y : np.array):
        """Fit the linear model to the data. Reshape the data into the desired shape to avoid any errors."""
        #convert the data into matrix form
        A = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        #perform the matrix calculations
        A_t = np.transpose(A)
        self.coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_t, A)), A_t), y)

    def predict(self, X : np.array) -> np.array:
        """Returns a vector containing the prediction of the model."""
        return np.matmul(np.append(np.ones((X.shape[0], 1)), X, axis=1), self.coeff)

    def score(self, X : np.array, y: np.array) -> float:
        """Returns the accuracy of the current state of the model. Uses the Metrics.accuracy_score method."""
        return accuracy_score(y, self.predict(X))
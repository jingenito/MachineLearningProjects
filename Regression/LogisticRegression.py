import numpy as np 

class LogisticRegression:
    """A class that represents a logistic regression model."""

    def __init__(self):
        #initialize
        self.coeff = []

    def get_params(self) -> np.array:
        """Returns the parameters of the model."""
        return self.coeff

    def fit(self, X : np.array, y : np.array):
        """Fit the logistic model to the data. Reshape the data into the desired shape to avoid any errors."""
        #convert the data into matrix form, note that we need a column of ones inserted at the beginning
        #to make the calculations work
        A = np.append(np.ones((X.size, 1)), X, axis=1)
        #it will be convenient to store the transpose for later
        A_t = np.transpose(A)
        self.coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_t, A)), A_t), y)

    def predict(self, X : np.array) -> np.array:
        """Returns a vector containing the predicted log-odds of the input data."""
        return np.matmul(np.append(np.ones((X.size, 1)), X, axis=1), self.coeff)

    def predict_proba(self, X : np.array) -> np.array:
        """Returns a vector containing the predicted probabilities of the input data"""
        log_odds = np.matmul(np.append(np.ones((X.size, 1)), X, axis=1), self.coeff)
        return np.exp(log_odds)/(1+ np.exp(log_odds))
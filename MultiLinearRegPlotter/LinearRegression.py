import numpy as np

def get_gradient_at_m_k(X, Y, M, b, k) :
    diff, N = 0, X.shape[0]
    for i in range(N) :
        x_val, y_val, y_hat = X[i, k], Y[i], b
        for j in range(X.shape[1]) :
            y_hat += M[j] * X[i, j]
        diff += x_val * (y_val - y_hat)
    return (-2/N) * diff

def get_gradient_at_b(X, Y, M, b) :
    diff, N = 0, X.shape[0]
    for i in range(N) :
        y_val, y_hat = Y[i], b
        for j in range(X.shape[1]) :
            y_hat += M[j] * X[i, j]
        diff += y_val - y_hat
    return (-2/N) * diff

def get_gradient_step(X, Y, M, b, learn_rate) :
    b_grad = get_gradient_at_b(X, Y, M, b)
    M_grad_step = np.zeros(X.shape[1])
    for k in range(X.shape[1]) :
        M_grad_step[k] = M[k] - learn_rate * get_gradient_at_m_k(X, Y, M, b, k)
    return (M_grad_step, b - learn_rate * b_grad)

def gradient_descent(X, Y, learn_rate, num_iterations) :
    M, b = np.ones(X.shape[0]), 0 #initial guess
    for i in range(num_iterations) :
        M, b = get_gradient_step(X, Y, M, b, learn_rate)
    return (M, b)

class LinearRegression:

    def __init__(self) :
        #initial guesses
        self.coeff, self.intercept = [], 0
        #Set N and learn rate
        self.num_iterations, self.learn_rate = 1000, 0.0001

    def fit(self, X, Y) :
        self.coeff, self.intercept = gradient_descent(X, Y, self.learn_rate, self.num_iterations)

    def predict(self, X) :
        return self.intercept + np.multiply(X, self.coeff)




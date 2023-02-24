def get_gradient_at_m(X, Y, m, b) :
    diff, N = 0, len(X)
    for i in range(N) :
        x_val, y_val = X[i], Y[i]
        diff += x_val * (y_val - ((m * x_val) + b))
    return (-2/N) * diff

def get_gradient_at_b(X, Y, m, b) :
    diff, N = 0, len(X)
    for i in range(N) :
        diff += Y[i] - ((m * X[i]) + b)
    return (-2/N) * diff

def get_gradient(X, Y, m, b) :
    return (get_gradient_at_m(X,Y,m,b), get_gradient_at_b(X,Y,m,b))

def get_gradient_step(X, Y, m, b, learn_rate) :
    gradient_m, gradient_b = get_gradient(X,Y,m,b)
    return (m - learn_rate * gradient_m, b - learn_rate * gradient_b)

def gradient_descent(X, Y, learn_rate, num_iterations) :
    m, b = 1, 0 #initial guess
    for i in range(num_iterations) :
        m, b = get_gradient_step(X, Y, m, b, learn_rate)
    return (m, b)

class SlowLinearRegression :
    def __init__(self) :
        #initial guesses
        self.coeff, self.intercept = 1, 0
        #Set N and learn rate
        self.num_iterations, self.learn_rate = 1000, 0.0001

    def fit(self, X, Y) :
        self.coeff, self.intercept = gradient_descent(X, Y, self.learn_rate, self.num_iterations)
    
    def predict(self, X) :
        return list(map(lambda x: (self.coeff*x) + self.intercept, X))
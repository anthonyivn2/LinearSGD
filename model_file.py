import numpy as np


class LinRegSGDModel:

    def __init__(self, alpha=0.1, epochs=10, shuffle=True,
                 random_state=None, regularization=None,
                 regularization_term=0):
        self.alpha = alpha
        self.b = 10
        self.a = alpha * self.b
        self.epochs = epochs
        self.shuffle = shuffle
        self.total_data = 0
        self.cost = []
        self.regularization = regularization
        self.regularization_term = regularization_term
        self.first_run = True
        self.random_state = random_state

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        self.total_data += len(self.y)
        self.y_rows = len(self.y)
        self.checkFirstRun()
        self.X = self.addBias(X)
        for epoch in range(self.epochs):
            if self.shuffle:
                self.shuffleData()
            cost = []
            for row in range(self.y_rows):
                cost.append(self.updateWeight(row, epoch))
            average_cost = sum(cost) / self.y_rows
            self.cost.append(average_cost)

    def predict(self, X):
        return np.dot(self.addBias(X), self.weights)

    def addBias(self, data):
        return np.concatenate((np.ones([data.shape[0], 1]), data), axis=1)

    def shuffleData(self):
        permutation = np.random.permutation(len(self.y))
        self.X = self.X[permutation]
        self.y = self.y[permutation]

    def checkFirstRun(self):
        if self.first_run:
            self.initializeWeight()
            self.first_run = False

    def initializeWeight(self):
        self.weights = np.random.randn(self.X.shape[1] + 1)

    def newAlpha(self, epoch):
        self.alpha = self.a / (self.b + epoch)

    def updateWeight(self, row_number, epoch):
        outcome = np.dot(self.X[row_number, :], self.weights)
        gradient, error = self.errorAndGradientCalculation(outcome, row_number)
        self.newAlpha(epoch)
        self.weights -= self.alpha * gradient
        cost_value = 0.5 * (error**2)
        return cost_value

    def errorAndGradientCalculation(self, outcome, row_number):
        error = outcome - self.y[row_number]
        if self.regularization == 'l2':
            return(
                self.l2RegularizationGradientCalculation(error, row_number),
                error)
        elif self.regularization == 'l1':
            return(
                self.l1RegularizationGradientCalculation(error, row_number),
                error)
        else:
            return(
                self.NoRegularizationGradientCalculation(error, row_number),
                error)

    def l2RegularizationGradientCalculation(self, error, row_number):
        return (np.dot(error, self.X[row_number, :]) +
                (self.regularization_term * self.weights / self.y_rows))

    def l1RegularizationGradientCalculation(self, error, row_number):
        return (np.dot(error, self.X[row_number, :]) +
                (self.regularization_term / self.y_rows *
                (np.abs(self.X[row_number, :]) / self.X[row_number, :])))

    def NoRegularizationGradientCalculation(self, error, row_number):
        return np.dot(error, self.X[row_number, :])

    def rSquared(self, target, prediction):
        mean = sum(target) / len(target)
        return 1 - (sum((prediction - target)**2) / sum((target - mean)**2))

    def RMSE(self, target, prediction):
        return (1/len(target) * sum((target - prediction)**2))**(1/2)

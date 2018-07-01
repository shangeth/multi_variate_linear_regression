import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

class gradient_descent_multi_linear_regression:

    def __init__(self,learning_rate=0.01,num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def computeCost(self,X, y, theta):
        tobesummed = np.power(((X @ theta.transpose()) - y), 2)
        return np.sum(tobesummed) / (2 * len(X))

    def gradientDescent(self,X, y, theta, iters, alpha):
        cost = np.zeros(iters)
        # print(X.shape)
        # print(np.array(y).shape)
        # print(theta.transpose().shape)
        # print((X @ theta.transpose()).shape)
        # print(np.subtract(np.array(X @ theta.transpose()),np.array(y)).shape)

        # print(((X @ theta.transpose()) - y).shape)
        # print("done")
        for i in range(iters):
            theta = theta - (alpha / float(len(X))) * np.sum(X * (X @ theta.transpose() - y), axis=0)
            cost[i] = self.computeCost(X, y, theta)
            print(str(i),"------",theta)

        return theta, cost

    def fit(self,X,y):
        self.X_ = X
        self.y = y
        self.theta = np.zeros([1, len(X[0])])
        g, cost = self.gradientDescent(X, y, self.theta, self.num_iterations, self.learning_rate)
        self.cost = cost
        self.theta = g
        finalCost = self.computeCost(X, y, g)


    def predict(self,X_pred,ymean,ystd,xmean,xstd):
        X_pred = np.array(X_pred)
        X_pred = (X_pred-xmean)/xstd
        print(X_pred,X_pred.shape)
        ones = np.ones([X_pred.shape[0], 1])
        X_pred = np.concatenate((ones, X_pred), axis=1)
        y_pred = X_pred @ self.theta.transpose()
        y_pred = y_pred * ystd + ymean
        print("prediction = ",y_pred)

    def squared_error(self,ys_orig, ys_line):
        sum = 0
        for i in range(len(ys_line)):
            sum += (ys_line[i]-ys_orig[i])**2
        return sum

    def coefficient_of_determination(self,ys_orig, ys_line):
        y_mean_line = [ys_orig.mean() for y in ys_orig]
        squared_error_regr = self.squared_error(ys_orig, ys_line)
        squared_error_y_mean = self.squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr / squared_error_y_mean)

    def r_squared(self,X,y):
        return self.coefficient_of_determination(np.array(y), (X @ self.theta.transpose()) )


    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.num_iterations), self.cost, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title('Error vs. Training Epoch')
        plt.show()





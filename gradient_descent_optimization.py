import pandas as pd
from sklearn.cross_validation import train_test_split
from Linear_Regression_Gradient_Descent import gradient_descent_multi_linear_regression
import numpy as np
from numpy import reshape
from sklearn.metrics import accuracy_score




def run():
    df = pd.read_csv("data.csv")
    y_mean = df[df.columns[-1]].mean()
    y_std = df[df.columns[-1]].std()
    print(y_std,y_mean)
    X_mean =df.drop("calories",axis=1).values.mean()
    X_std = df.drop("calories",axis=1).values.std()

    df = (df-df.mean())/df.std()
    y = df[df.columns[-1]]
    X = df.drop("calories",axis=1).values




    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    y = y.values.reshape((y.values.shape[0], 1))








    clf = gradient_descent_multi_linear_regression()
    clf.fit(X,y)
    clf.plot()
    clf.predict([[38]],y_mean,y_std,X_mean,X_std)

    r = clf.r_squared(X,y)
    print("r_square value = ",float(r))



if __name__=="__main__":
    run()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def import_data():
    ## remember change it
    train = pd.read_csv("task1b/train.csv")
    train_data = train.values
    
    X_train = train_data[:,2:]
    Y_train = train_data[:,1]
    return X_train,Y_train

def main():

    X_train, Y_train = import_data()
    
    # compute features
    # phi1-5 is already in X_train
    quadratic = np.square(X_train)
    exp = np.exp(X_train)
    cosine = np.cos(X_train)
    constant = np.ones(700)
    constant = constant.reshape((700,-1))

    # shape of X_features: (700, 21)
    X_features = np.hstack((X_train, quadratic, exp, cosine, constant))

    #linear regression
    regr = LinearRegression(fit_intercept=False)
    regr = regr.fit(X_features, Y_train)

    #Get coefficient
    coef = regr.coef_

    #write in solution.csv
    output = pd.DataFrame(data=coef, columns=None)
    output.to_csv("task1b/solution.csv", index = False, header=False)


if __name__ == "__main__":
    main()
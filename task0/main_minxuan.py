import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def import_data():
    train = pd.read_csv("train.csv")
    X_train = train.values
    y_train = X_train[:,1]
    X_train = X_train[:,2:]

    test = pd.read_csv("test.csv")
    X_test = test.values
    X_test = X_test[:,1:]
    return X_train, y_train, X_test

def generate_solution(y_pred):
    filename = 'solution.csv'
    header = ['Id', 'y']
    ids = np.linspace(10000.0, 11999.0, num=2000)
    ids = ids.reshape((2000,))
    body = np.vstack([ids,y_pred])
    data = pd.DataFrame(body.transpose(), columns = header)
    data.to_csv(filename, index=False)


X_train, y_train, X_test = import_data()
reg = LinearRegression()
reg = reg.fit(X_train, y_train)
y_test = reg.predict(X_test)

## check mse
means = np.mean(X_test, axis=1)

RMSE = mean_squared_error(means, y_test) ** 0.5
print(RMSE)

generate_solution(y_test)
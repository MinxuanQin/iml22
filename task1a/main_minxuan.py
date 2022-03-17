import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def import_data():
    ## remember change it
    train = pd.read_csv("task1a/train.csv")
    train_data = train.values
    
    X_train = train_data[:,1:]
    Y_train = train_data[:,0]
    return X_train,Y_train

def main():
    X_train, Y_train = import_data()
    ## model selector
    kf = KFold(n_splits=10, shuffle=True)

    ## parameter
    lamb = np.array([0.1, 1, 10, 100, 200])
    rse = []
    intermediate = []
    
    for i in lamb:
        ## model
        ridge = Ridge(alpha=i)
        ## k fold
        for train_index, test_index in kf.split(X_train, Y_train):
            x_train, x_test = X_train[train_index], X_train[test_index]
            y_train, y_test = Y_train[train_index], Y_train[test_index]
            ridge.fit(x_train, y_train)
            y_pred = ridge.predict(x_test)
            intermediate.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        ## append rse
        mses = np.array(intermediate)
        rse.append(np.mean(mses))
    

    ## solution.csv
    
    rse = np.array(rse)
    output = pd.DataFrame(data=rse, columns=None)
    output.to_csv("task1a/solution.csv", index = False, header=False)
    

if __name__ == "__main__":
    main()
import pandas as pd
from typing import List, Dict
from sklearn.datasets import load_iris
import numpy as np

def scratch_linearreg_pred(X:pd.DataFrame,
                           m:int,
                           b:int) -> List:
    producto_punto = np.dot(X.to_numpy(dtype='float', na_value=0), m) 
    predicted_y = producto_punto + b
    return predicted_y    

def scratch_linearreg_fit(X: pd.DataFrame, 
                          y: pd.DataFrame,
                          learning_rate:int = .01,
                          train_iterations:int = 100000) -> Dict:
    X = X.to_numpy(dtype='float', na_value=0)                                 
    y = y.to_numpy(dtype='float', na_value=0).reshape(len(y))        

    shape = X.shape
    X_size = shape[0]
    X_cols = shape[1]

    m = np.zeros(X_cols)  # slope, weight
    b = 0                 # intercept, bias
    while (train_iterations != 1):   
        denominador =  (1 / X_size)
        loss = (np.dot(X, m) + b) - y
        m = m - (learning_rate * denominador * np.dot(X.T, loss))
        b = b - (learning_rate * denominador * np.sum(loss))
        
        train_iterations = train_iterations - 1 
    return {"coefficients":m, "intercept":b}

def example_scratch_linearreg_iris():
    print("This test Linear Reg using iris dataset ")
    
    df = load_iris(as_frame=True).frame
    df.columns = ["sepal length","sepal width","petal length","petal width", "class"]
    
    df_x = df[["sepal length","sepal width","petal length","petal width"]]
    df_y = df[["class"]]    
    results = scratch_linearreg_fit(df_x, df_y)
    print("Coefficients: ",results["coefficients"])
    print("Intercept: ", results["intercept"])
    print(scratch_linearreg_pred(df_x, results["coefficients"], results["intercept"]))

if __name__ == '__main__':
    example_scratch_linearreg_iris()
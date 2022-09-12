from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp

def ridge_lasso(X: pd.DataFrame, 
               y: pd.DataFrame,
               alpha:int = .3) -> Dict:
    X = X.to_numpy(dtype='float', na_value=0)                                 
    y = y.to_numpy(dtype='float', na_value=0).reshape(len(y))        

    ridge = Ridge(alpha=.3)
    ridge.fit(X, y)

    lasso = Lasso(alpha=.3)
    lasso.fit(X, y)    
    return {"ridge coefficients":ridge.coef_, 
            "ridge intercept":ridge.intercept_, 
            "ridge model": ridge,
            "lasso coefficients":lasso.coef_, 
            "lasso intercept":lasso.intercept_, 
            "lasso model": lasso}
    
def linearreg_pred(X:pd.DataFrame,
                   model: LinearRegression,
                   binarize:bool = False,
                   threshold:int =  0.5) -> List:
    X = X.to_numpy(dtype='float', na_value=0)             
    preds = model.predict(X)                    
    return  np.where(preds>threshold, 1, 0) if binarize else preds

def scratch_linearreg_pred(X:pd.DataFrame,
                           m:int,
                           b:int,
                           binarize:bool = False,
                           threshold:int =  0.5) -> List:
    producto_punto = np.dot(X.to_numpy(dtype='float', na_value=0), m) 
    predicted_y = producto_punto + b
    return np.where(predicted_y>threshold, 1, 0) if binarize else predicted_y    

def scikit_linearreg_fit(X: pd.DataFrame, 
                         y: pd.DataFrame,
                         learning_rate:int = .01,
                         train_iterations:int = 100000) -> Dict:
    X = X.to_numpy(dtype='float', na_value=0)                                 
    y = y.to_numpy(dtype='float', na_value=0).reshape(len(y))        

    m1 = LinearRegression().fit(X, y)
    return {"coefficients":m1.coef_, 
            "intercept":m1.intercept_, 
            "model": m1}
    
def scikit_bias_variance(X: pd.DataFrame, 
                         y: pd.DataFrame,
                         model: LinearRegression,
                         test_size:int = .33,
                         random_state:int = 43,
                         num_rounds:int =  200,
                         loss:str = 'mse') -> Dict:
    X = X.to_numpy(dtype='float', na_value=0)                                 
    y = y.to_numpy(dtype='float', na_value=0).reshape(len(y))        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss=loss, num_rounds=num_rounds, random_seed=random_state)
    
    return {"mse_bias":mse, 
            "bias":bias, 
            "var": var}

def example_scikit_linearreg_iris():
    print("This test Scikit Linear Reg using iris dataset")
    
    df = load_iris(as_frame=True).frame
    df.columns = ["sepal length","sepal width","petal length","petal width", "class"]
    
    df_x = df[["sepal length","sepal width","petal length","petal width"]]
    df_y = df[["class"]]    
    
    results = scikit_linearreg_fit(df_x, df_y)
    
    print("Coefficients: ", results["coefficients"])
    print("Intercept:    ", results["intercept"])
    print("Scratch preds:", scratch_linearreg_pred(df_x, results["coefficients"], results["intercept"]))
    print("Scikit preds: ", linearreg_pred(df_x, results["model"]))
    
if __name__ == '__main__':
    example_scikit_linearreg_iris()

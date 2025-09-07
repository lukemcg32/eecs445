"""
EECS 445 - Introduction to Machine Learning
HW1 Q7 Logistic Regression
"""

import numpy as np
from helper import load_data

def sigmoid(z):
    """ 
    Implements the sigmoid function..
    Args:
        z: A scalar or numpy array of any size
    """
    if isinstance(z, np.ndarray):
        result = list(z)
        return np.ndarray([1 if x > 0 else 0 for x in result])
    
    else:
        return 1 if z > 0 else 0


def logistic_stochastic_gradient_descent(X, y, lr=0.0001):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for logistic regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of epochs == 10,000
    
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        lr: the learning rate for the algorithm
    
    Returns:
        theta: np.array, shape (d+1,) including the offset term.
    """
    n, d = X.shape
    theta = np.zeros(d + 1)

    Xbar = np.hstack([np.ones((n, 1)), X])

    for _ in range(10000):

        # one pass over the data, in order
        for xi, yi in zip(Xbar, y):
            score = theta @ xi
            gradient = -1 * yi * xi * sigmoid(-yi * score)
            theta -= lr * gradient

    return theta


    
    # TODO: Implement SGD. Train for 10,000 epochs
    return theta

def stochastic_newtons_method(X, y, lr=0.0001):
    """
    Implements the Stochastic Newton's Method algorithm for logistic regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of epochs == 1,000

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        lr: the learning rate for the algorithm

    Returns:
        theta: np.array, shape (d+1,) including the offset term.
    """
    n, d = X.shape
    theta = np.zeros(d + 1)

    # TODO: Implement Stochastic Newton's Method. Train for 1,000 epochs
    return theta

def main(fname):
    X, y = load_data(fname)
    theta_SGD = logistic_stochastic_gradient_descent(X, y)
    print("SGD Theta: ", theta_SGD)
 

    theta_Newtons = stochastic_newtons_method(X, y)
    print("Newton's Method Theta: ", theta_Newtons)


if __name__ == '__main__':
    main('dataset/q7.csv')

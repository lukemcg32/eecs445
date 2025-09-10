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
    # handles scalars
    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-z))


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
    step = lr

    Xbar = np.hstack([np.ones((n, 1)), X])

    for _ in range(10000):

        # no shuffle
        for xi, yi in zip(Xbar, y):
            theta_dot_x = np.dot(theta, xi)
            gradient = -1 * yi * xi * sigmoid(-yi * theta_dot_x)
            theta -= step * gradient

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
    step = lr

    Xbar = np.hstack([np.ones((n, 1)), X])
    eps = 1e-10

    #epochs = 1000
    epochs = 2000

    for _ in range(epochs):

        # no shuffle
        for xi, yi in zip(Xbar, y):
            margin = np.dot(theta, xi)
            sigmoid_signed = sigmoid(yi * margin)

            beta = -1 * yi * (1.0 - sigmoid_signed)
            alpha = sigmoid_signed * (1.0 - sigmoid_signed)

            # add eps to avoid 0 denom
            newton_step = (beta / (alpha * (xi @ xi) + eps) ) * xi
            theta -= step * newton_step


    return theta




def main(fname):

    print('\n')

    X, y = load_data(fname)
    theta_SGD = logistic_stochastic_gradient_descent(X, y)
    print("SGD Theta: ", theta_SGD)
    print('\n')
 

    theta_Newtons = stochastic_newtons_method(X, y)
    print("Newton's Method Theta: ", theta_Newtons)
    print('\n')


if __name__ == '__main__':
    main('dataset/q7.csv')

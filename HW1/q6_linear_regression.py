"""
EECS 445 - Introduction to Maching Learning
HW1 Q6 Linear Regression Optimization Methods)
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
from helper import load_data
import time

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        theta: np.array, shape (d,). Specifies an (d-1)^th degree polynomial
        
    Returns:
        The squared loss for the given data and parameters
    """
    n, _ = X.shape
    r = X @ theta - y
    return 0.5 * np.dot(r, r) / n

def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implement the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the following stopping criteria together: number of iterations >= 1e6 or |new_loss − prev_loss| <= 1e−10

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
    
    Returns:
        theta: np.array, shape (d,)
    """
    _, d = X.shape
    theta = np.zeros(d)

    eps = 1e-10
    max_iter = 1e6
    n_iter = 0

    step = learning_rate
    prev_loss = np.inf
    new_loss = calculate_squared_loss(X, y, theta)



    while (n_iter < max_iter) and (abs(new_loss - prev_loss) > eps):
        n_iter += 1
        grad = []

        for xt, yt in zip(X, y):
            pred = np.dot(theta, xt)
            g = (pred - yt) * xt
            grad.append(g)
        
        gradient = np.mean(grad, axis=0)
        theta -= gradient * step

        prev_loss = new_loss
        new_loss = calculate_squared_loss(X, y, theta)

    print("Learning rate:", learning_rate, "\t\t\tNum iterations:", n_iter)
    return theta




def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implement the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the following stopping criteria together: number of iterations >= 1e6 or |new_loss − prev_loss| <= 1e−10
    
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        learning_rate: the learning rate for the algorithm
    
    Returns:
        theta: np.array, shape (d,)
    """
    _, d = X.shape
    theta = np.zeros(d)
    adaptive = (learning_rate == 'adaptive')

    eps = 1e-10
    max_iter = 1e6
    n_iter = 0
    epochs = 0

    step = learning_rate
    prev_loss = np.inf
    new_loss = calculate_squared_loss(X, y, theta)

    if adaptive:
        step = 0.01
    else:
        step = 0.01 if learning_rate == 0 else float(learning_rate)

    while (n_iter < max_iter) and (abs(new_loss - prev_loss) > eps):
        epochs += 1

        # simply decrementing our learning rate as we go...gradually approaches 0
        if adaptive:
            step /= (epochs + 1)


        for xt, yt in zip(X, y):
            pred = np.dot(theta, xt)
            gradient = (pred - yt) * xt

            theta -= step * gradient
            n_iter += 1

            if n_iter >= max_iter:
                break

        prev_loss = new_loss
        new_loss = calculate_squared_loss(X, y, theta)

    print("Learning rate:", learning_rate, "\t\t\tNum iterations:", n_iter)
    return theta



def ls_closed_form_optimization(X, y):
    """
    Implement the closed form solution for least squares regression.

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)

    Returns:
        theta: np.array, shape (d,)
    """
    # piazza @85

    return np.linalg.pinv(X) @ y

def main(fname_train):
    # TODO: This function should contain all the code you implement to complete question 6.

    X_train, y_train = load_data(fname_train)
    
    # Appending a column of constant ones to the X_train matrix to make X_train the same dimensions as theta.
    # The term multiplied by theta_0 is x^0 = 1 (theta_0 is a constant), which is why the column contains only ones.
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    print("Closed-form:")
    theta_cf = ls_closed_form_optimization(X_train, y_train)
    loss_cf = calculate_squared_loss(X_train, y_train, theta_cf)
    print("Squared Loss:", loss_cf)
    print("Theta", np.linalg.norm(theta_cf))
    print("\n")


    print("Gradient Decent with LR=0.01:")
    theta_gd = ls_gradient_descent(X_train, y_train, learning_rate=0.01)
    loss_gd = calculate_squared_loss(X_train, y_train, theta_gd)
    print("Squared Loss:", loss_gd)
    print("Theta", np.linalg.norm(theta_gd))
    print("\n")

    print("Gradient Decent with LR=0.05:")
    theta_gd2 = ls_gradient_descent(X_train, y_train, learning_rate=0.05)
    loss_gd2 = calculate_squared_loss(X_train, y_train, theta_gd2)
    print("Squared Loss:", loss_gd2)
    print("Theta", np.linalg.norm(theta_gd2))
    print("\n")



    print("Stochastic Gradient Decent with LR=0.01:")
    theta_sgd = ls_stochastic_gradient_descent(X_train, y_train, learning_rate=0.01)
    loss_sgd = calculate_squared_loss(X_train, y_train, theta_sgd)
    print("Squared Loss:", loss_sgd)
    print("Theta", np.linalg.norm(theta_sgd))
    print("\n")

    print("Stochastic Gradient Decent with LR=0.05:")
    theta_sgd1 = ls_stochastic_gradient_descent(X_train, y_train, learning_rate=0.05)
    loss_sgd1 = calculate_squared_loss(X_train, y_train, theta_sgd1)
    print("Squared Loss:", loss_sgd1)
    print("Theta", np.linalg.norm(theta_sgd1))
    print("\n")

    print("Stochastic Gradient Decent with LR=\"adaptive\":")
    theta_sgd2 = ls_stochastic_gradient_descent(X_train, y_train, learning_rate='adaptive')
    loss_sgd2 = calculate_squared_loss(X_train, y_train, theta_sgd2)
    print("Squared Loss:", loss_sgd2)
    print("Theta", np.linalg.norm(theta_sgd2))
    print("\n")

    print("Done!")



if __name__ == '__main__':
    main("dataset/q6.csv")

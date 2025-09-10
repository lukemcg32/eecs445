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
        step = 0.25    # best results
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



# used genAI to generate this table
def one_click_summary(X_train, y_train):
    configs = [
        ("GD", 1e-4, ls_gradient_descent),
        ("GD", 1e-3, ls_gradient_descent),
        ("GD", 1e-2, ls_gradient_descent),
        ("GD", 1e-1, ls_gradient_descent),
        ("SGD", 1e-4, ls_stochastic_gradient_descent),
        ("SGD", 1e-3, ls_stochastic_gradient_descent),
        ("SGD", 1e-2, ls_stochastic_gradient_descent),
        ("SGD", 1e-1, ls_stochastic_gradient_descent),
        ("SGD", "adaptive", ls_stochastic_gradient_descent),
        ("Closed form", None, ls_closed_form_optimization),
    ]

    rows = []
    for algo, lr, fn in configs:
        t0 = time.process_time()
        if algo == "Closed form":
            theta = fn(X_train, y_train)
        else:
            theta = fn(X_train, y_train, learning_rate=lr)
        runtime = time.process_time() - t0

        loss = calculate_squared_loss(X_train, y_train, theta)
        t0_hat = float(theta[0])
        t1_hat = float(theta[1]) if theta.shape[0] > 1 else float("nan")
        rows.append((algo, lr, t0_hat, t1_hat, runtime, loss))

    print("\nAlgorithm        η           θ0           θ1        CPU time(s)         Loss")
    print("-"*90)
    for algo, lr, t0_hat, t1_hat, rt, loss in rows:
        eta_str = f"{lr:>5.0e}" if isinstance(lr, float) else (lr if lr is not None else "-")
        print(f"{algo:<12} {eta_str:>8}   {t0_hat:> .6f}   {t1_hat:> .6f}   {rt:>12.6f}   {loss:> .6e}")


def main(fname_train):
    # TODO: This function should contain all the code you implement to complete question 6.

    X_train, y_train = load_data(fname_train)
    
    # Appending a column of constant ones to the X_train matrix to make X_train the same dimensions as theta.
    # The term multiplied by theta_0 is x^0 = 1 (theta_0 is a constant), which is why the column contains only ones.
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    print("Closed-form:")
    t11 = time.process_time()
    theta_cf = ls_closed_form_optimization(X_train, y_train)
    t1 = t11 - time.process_time()
    loss_cf = calculate_squared_loss(X_train, y_train, theta_cf)
    print("Squared Loss:", loss_cf)
    print("Theta", np.linalg.norm(theta_cf))
    print("\n")

    
    print("Gradient Decent with LR=0.01:")
    t21 = time.process_time()
    theta_gd = ls_gradient_descent(X_train, y_train, learning_rate=0.01)
    t2 = t21 - time.process_time()
    loss_gd = calculate_squared_loss(X_train, y_train, theta_gd)
    print("Squared Loss:", loss_gd)
    print("Theta", np.linalg.norm(theta_gd))
    print("\n")

    print("Gradient Decent with LR=0.05:")
    t31 = time.process_time()
    theta_gd2 = ls_gradient_descent(X_train, y_train, learning_rate=0.05)
    t3 = t31 - time.process_time()
    loss_gd2 = calculate_squared_loss(X_train, y_train, theta_gd2)
    print("Squared Loss:", loss_gd2)
    print("Theta", np.linalg.norm(theta_gd2))
    print("\n")



    print("Stochastic Gradient Decent with LR=0.01:")
    t41 = time.process_time()
    theta_sgd = ls_stochastic_gradient_descent(X_train, y_train, learning_rate=0.01)
    t4 = t41 - time.process_time()
    loss_sgd = calculate_squared_loss(X_train, y_train, theta_sgd)
    print("Squared Loss:", loss_sgd)
    print("Theta", np.linalg.norm(theta_sgd))
    print("\n")

    print("Stochastic Gradient Decent with LR=0.05:")
    t51 = time.process_time()
    theta_sgd1 = ls_stochastic_gradient_descent(X_train, y_train, learning_rate=0.05)
    t5 = t51 - time.process_time()
    loss_sgd1 = calculate_squared_loss(X_train, y_train, theta_sgd1)
    print("Squared Loss:", loss_sgd1)
    print("Theta", np.linalg.norm(theta_sgd1))
    print("\n")

    print("Stochastic Gradient Decent with LR=\"adaptive\":")
    t61 = time.process_time()
    theta_sgd2 = ls_stochastic_gradient_descent(X_train, y_train, learning_rate='adaptive')
    t6 = t61 - time.process_time()
    loss_sgd2 = calculate_squared_loss(X_train, y_train, theta_sgd2)
    print("Squared Loss:", loss_sgd2)
    print("Theta", np.linalg.norm(theta_sgd2))
    print("\n")

    print("Done!")


    # one stop shop
    print("------------------------------------------------------------------------\n\n")
    one_click_summary(X_train, y_train)
    




if __name__ == '__main__':
    main("dataset/q6.csv")

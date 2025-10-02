"""
EECS 445 - Introduction to Maching Learning
HW2 Q1 Linear Regression Optimization Methods
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from q1_helper import load_data

def calculate_MS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        theta: np.array, shape (d,). Specifies an (d-1)^th degree polynomial

    Returns:
        E_ms: float. The root mean square error as defined in the assignment.
    """
    y_pred = X.dot(theta)
    E_ms = np.mean(np.square(y_pred - y))
    return E_ms


def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. x -> [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (n, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (n, M+1)
    """
    n, _ = X.shape
    Phi = np.zeros((n, M+1))

    # TODO: (a) - Implement this function
    arrM = np.arange(M+1)
    Phi = X ** arrM

    return Phi


def closed_form_optimization(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: (c) - Modify implementation to incorporate L2-regularization
    # setting gradient = 0 ... theta = (XTX/n + lambda*identity)^-1 @ (XTy / n)

    n, d = X.shape
    # theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    identity = np.eye(d)

    A = (X.T @ X) / n
    A = A + (reg_param * identity)

    b = (X.T @ y) / n

    # A@theta = b --> theta = A^-1 @ b
    theta = np.linalg.solve(A, b)

    return theta





def part_1(fname_train, fname_validation):
    print("=========== Q1 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # (b) OVERFITTING

    errors_train = np.zeros((11,))
    errors_validation = np.zeros((11,))

    # TODO: Add your part (b) code here
    for M in range(11):
        Phi_train = generate_polynomial_features(X_train, M)
        Phi_val   = generate_polynomial_features(X_validation, M)

        theta = closed_form_optimization(Phi_train, y_train, reg_param=0)

        errors_train[M] = calculate_MS_Error(Phi_train, y_train, theta)
        errors_validation[M] = calculate_MS_Error(Phi_val, y_validation, theta)
    

    plt.plot(errors_train,'-or',label='Train')
    plt.plot(errors_validation,'-ob', label='Validation')
    plt.xlabel('M')
    plt.ylabel('$MSE$')
    plt.title('Question 1(b)')
    plt.legend(loc=1)
    plt.xticks(np.arange(0, 11, 1))
    plt.savefig('q1b.png', dpi=200)
    plt.close()


    # (c) REGULARIZATION

    errors_train = np.zeros((10,))
    errors_validation = np.zeros((10,))
    M = 10
    L = np.append([0], 10.0 ** np.arange(-8, 1))
    # L = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    # same thing I'm pretty sure ^^^

    # TODO: Add you part (c) code here

    # only need to build these once because we aren't changing anything but lambda
    Phi_tr = generate_polynomial_features(X_train, M)
    Phi_va = generate_polynomial_features(X_validation, M)

    for i, lam in enumerate(L):
        theta = closed_form_optimization(Phi_tr, y_train, reg_param=lam)
        errors_train[i] = calculate_MS_Error(Phi_tr, y_train, theta)
        errors_validation[i] = calculate_MS_Error(Phi_va, y_validation, theta)
    
    plt.figure()
    plt.plot(L, errors_train, '-or', label='Train')
    plt.plot(L, errors_validation, '-ob', label='Validation')
    plt.xscale('symlog', linthresh=1e-8)
    plt.xlabel('$\lambda$')
    plt.ylabel('$MSE$')
    plt.title('Question 1(c)')
    plt.legend(loc=2)
    plt.savefig('q1c.png', dpi=200)
    plt.close()

    print("Done!")


def main(fname_train, fname_validation):
    part_1(fname_train, fname_validation)

if __name__ == '__main__':
    main("q1_data/q1_train.csv", "q1_data/q1_validation.csv")

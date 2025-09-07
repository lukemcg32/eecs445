"""
EECS 445 - Introduction to Machine Learning
HW1 Q5 Perceptron Algorithm with Offset
"""

import numpy as np
from helper import load_data

def all_correct(X, y, theta, b):
    """
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        theta: np.array, shape (d,), normal vector of decision boundary
        b: float, offset

    Returns true if the linear classifier specified by theta and b correctly classifies all examples
    """

    for i in range(len(X)):
        for i in range(len(X)):
            pred = np.dot(theta, X[i]) + b
            if y[i] * pred <= 0:
                return False
    
    return True



def perceptron(X, y):
    """
    Implements the Perception algorithm for binary linear classification.
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)

    Returns:
        theta: np.array, shape (d,)
        b: float
        alpha: np.array, shape (n,). 
            Misclassification vector, in which the i-th element is has the number of times 
            the i-th point has been misclassified)
    """
    n, d = X.shape
    theta = np.zeros((d,))
    b = 0.0
    alpha = np.zeros((n,))

    updated = True
    while updated:
        updated = False
        for i in range(n):
            margin = y[i] * (np.dot(theta, X[i]) + b)
            if margin <= 0:
                theta += y[i] * X[i]
                b += y[i]
                alpha[i] += 1
                updated = True

    return theta, b, alpha



def main(fname):
    X, y = load_data(fname)
    theta, b, alpha = perceptron(X, y)

    print("Done!")
    print("============== Classifier ==============")
    print("Theta: ", theta)
    print("b: ", b)

    print("\n")
    print("============== Alpha ===================")
    print("i \t Number of Misclassifications")
    print("========================================")
    for i in range(len(alpha)):
        print(i, "\t\t", alpha[i])
    print("Total Number of Misclassifications: ", np.sum(alpha))


if __name__ == '__main__':
    main('dataset/q5.csv')
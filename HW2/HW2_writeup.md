1.a.
*ss of code**

1.b.i.
*ss of q1b.png*

1.b.ii.
The polynomial of degree 4 or 5 best fits the data. Lower degrees (0 and 1) underfit, and higher degrees (M > 7) overfit

1.c.i.
*ss of code*

1.c.ii.
*ss of q1c.png*

1.c.ii.
The lambda value of 1e-6 has the lowest validation MSE and has a train MSE very close to zero. We see little change in the train MSE, so we should take what appears to be the smallest validation MSE.

2.a.
Frederick’s setup is better because it uses cross-validation within the training set to select hyperparameters, avoiding contaminating the test set. Nathan’s approach tunes hyperparameters directly on the test data, leading to overfitting to the test set and an overly optimistic estimate. Therefore, Frederick’s method is more likely to reflect true performance on unseen data.

2.b.
The best practice all three would agree on is to use cross-validation on the training data, select hyperparameters on the validation set, and then retrain on the full training set with the best hyperparameters from cross validation. Lastly, the model would be tested and report performance on the untouched test set. This would give the hyperparameters that generalize the best to unseen data and then train a model that has has been trained on all the data, leading to the best model and the most accurate performance report.


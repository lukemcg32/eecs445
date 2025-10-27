1.a.
* include ss of dataset.py code *

1.a.ii. 
Mean:   R: 125.55  G: 118.788  B: 94.766
Std:    R: 64.413  G: 61.379   B: 64.247

1.a.ii.
We only take the mean and standard deviation from the training data because that’s the only data the model is allowed to see while learning. This will insure that when we test against the validation and test sets that it is a real life prediction of how the model will generalize to unseen data.

1.b.
* include ss of og_vs_preprocessed.png *
We see a set of softer and blurrier images.

2.a.
Layer 0: 0 params
Layer 1: 
output channels * (weights per filter) + number of biases
16 * (3*5*5) + 16 = 1216 params
Layer 2: 0 params
Layer 3: 
output channels * (weights per filter) + number of biases
64 * (16*5*5) + 64 = 25664 params
Layer 4: 0 params
Layer 5: 
output channels * (weights per filter) + number of biases
8 * (64*5*5) + 8 = 12808 params
Layer 6: 2 * 32 + 2 = 66 params

Total learnable params = 39,754

2.b.
* include ss of target.py code *
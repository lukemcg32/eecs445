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

2.c. 
* include ss of train_common.py code *

2.d.
* include ss of train_cnn.py code *

2.e.
* include ss of train_common.py code *

2.f.i.
The validation loss wiggles because the model sees different mini-batches of data each epoch, and the randomness in which samples are grouped together can temporarily push the loss up even if training is going well overall. The optimizer also makes updates based only on the current batch rather than the entire dataset, which introduces noise into each step. Since the model starts from randomly initialized weights, early training is especially unstable and sensitive to batch composition. All of this randomness naturally leads to small fluctuations in the validation loss.

2.f.ii.
With patience=5, the model stopped at epoch 18. With patience=10, the model stopped at epoch 23. The two different stopping points appear to do about the same. Looking at the actual epochs though, the model with patience=10 goes 5 more epochs, which means all of the additional epochs past patience=5 are worse than our best validation accuracy. Therefore, I would say the model with patience=5 is better.

* inlcude ss of cnn_training_plot_patience=5.png *
* inlcude ss of cnn_training_plot_patience=10.png *

2.f.iii.
2 * 2 * 64 = 256

            Epoch    Training AUROC      Validation AUROC
8  filters:   13          1.0                  0.9975
64 filters:   8           1.0                  0.9959

The model with a 64 layer output from the third convolution layer is more complex and converges faster. Looking at the validation AUROC, it seems as though the 8 layer model does slightly better generalizing to unseen data. This adds up because the variance of the more complex model is expected to rise slightly as we add complexity, which is what we did here. This leads to slightly worse generalization which is shown in the results.

2.g.i.
            Training     Validation     Testing
Accuracy      1.0          0.9667         0.56
AUROC         1.0          0.9975         0.7668

2.g.ii.
We see some evidence of overfitting with respect to training vs validation accuracy and AUROC, but it doesn't seem to be doing too poorly. The models inability to effectively generalizer to unseen data in the tetsing accuracy and AUROC show the model's tendency to overfit.

2.g.iii. 
The positive labeled data has a red and black box in the upper right hand corner. It is possible that the model learned that the black box iin the upper right hand corner was the key indication of whether or not the dog in a collie and that the black and red boxes are not consistent against the two breeds in the test set.

3.a.
* include ss of source.py code *

3.b. 
* include ss of source_train.py code *

3.c. 
Epoch with lowest val loss = 1.7623 is epoch 9.
* include ss of 3c_source_training_plot_patience=10.png *

3.d. 
* include ss of 3d_confusion_matrix.png *
The breeds the model is most accurate are Samoyed, Great Dane, Dalmation, and Yorkshire Terrier. The least accurate are Miniature Poodle, Chihuahua, and Siberian Husky (worst). Looking into bad performance, it looks like our model misclassifies Miniature Poodles as Yorkies, Chihuahua as samoyeds, and is confused about Siberian Huskies but classifies them the most often as Saint Bernards.

Miniature poodles and yorkies are both small and can have similar coloring. Siberian Huskies and Saint Bernards are both large dogs but do not have many other similar traits. Chihuahuas and Samoyeds have very little resemblence, so our model doesn't seem to have learned the Chihuahua characteristics.

3.e.
* include ss of train_target.py *

3.f.
                    Train AUROC     Val AUROC       Test AUROC
Freeze all:             0.8688        0.8779            0.7764  
Freeze first two:       1.0           0.9868            0.776 
Freeze first:           1.0           0.9963            0.7548
Freeze none:            1.0           0.9989            0.7716
--------------------------------------------------------------
No pretraining:         1.0           0.9975            0.7668

From what we see above, freezing no layers, freezing two layers, and freezing all convolutional layers improved our models' test AUROC. This makes sense because if we only freeze the first layer, we hurt the model because it feeds in only low level edge detectors and does not provide much of the learned representation. Freezing the first two and all three convolutional layers gives the model more transfer representations and we see the best performance with these two (freezing all layers is the best). Freezing no layers also shows improvement because the model has full flexibility to learn and starts at the weights from the learned representations.

The observation that more than freezing two or all layers outperforms freezing no layers checks out because the model retrains some valueable weight information from the transfer because it is given so much flexibility.


4.

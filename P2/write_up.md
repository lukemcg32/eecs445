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


4.a.
If the [CLS] embedding were removed, we could still form a classification by applying an average pooling over all the patch embeddings output from the transformer encoder and then feed that pooled vector into the MLP head.

But, we see that this approach was tested by Dosovitskiy and was found to perform worse than using the [CLS] token because the token acts as a learnable aggregator that collects information from all patches through self-attention and learns to represent the entire image in a way optimized for classification. Without it the model relies on averaging and doesn't adaptively weight the important patches.

4.b.
The ViT would lose spatial awareness because the transformer would treat all patches as an unordered set. The model could detect what features exist in the image but not where they are actually located.

4.c.
* include ss from models/vit.py *

4.d.
* include ss from models/vit.py *

4.e. 
* include ss from models/vit.py *

4.f.
    i. 1 x D, so 16 learnable parameters

    ii. D_patch = 3 x (64/16) x (64/16) = 3 x 4 x 4 = 48

        weight matrix + bias = D_patch x D + D  = 48*16 + 16 = 784 learnable parameters

    iii.
        1. 2 * (weight + bias) = 2*2D = 2*2*16 = 64 learnable parameters

        2. total = heads + output projection = 1088 learnable parameters
           heads = 3h(D*(D/h) + (D/h)) = 3(D*D + D) = 816
           output projection = D*D + D = 272

        3. total = first + second = 1088 + 1040 = 2128 learnable parameters
           first = weights + biases = in * out + bias = 4*16*16 + 4*16 = 1088
           second = weights + biases = in * out + bias = 4*16*16 + 16 = 1040

        total = 2 * (64 + 1088 + 2128) = 6560 learnable parameters in single transformer block

    iv. D * number of classes + number of classes = 2*16 + 2 = 34 learnable paramters

    v. total = 16 + 784 + 6560 + 34 = 7394 learnable parameters

4.g.
* include ss of train_vit.py *

4.h.
epoch 14 best validation loss

Train AUROC:        0.992
Validation AUROC:   0.9819

* include ss of 4h_vit_training_plot_patience=5.png *

4.i.
Test AUROC: 0.6452

4.j.
The ViT does not perform as well as our CNN from section 2g in terms of test and validation AUROC. In fact the CNN does noticeably better with a test AUROC of 0.7668 compared to the 0.6452 test AUROC of our ViT. However, the ViT has far less paramters (7,394) compared to the CNN (39,754). This lead to much faster training. In the case where we had more data and limited computing power, the ViT would be a more efficient option. 



Challenge:

** Brain Storming **
- need to augment data to train up a sweet model on Colab GPUs
    - increases data size 4x and makes slightly more robust potentially?
    - maybe we can augment every other image or something so we don;t get too dependent on different formats
- need to make our CNN model larger (think VGG)
        - keep filter size 3 and stride 1 or same
        - ^^ ensures that we capture as much information as we possibly can while making our net deep ^^
        - maybe make some layers conv layers that don't downsample because we have such limited image sizes
- incorperate transformer/pretraining architecture
    - transfer learning will be faster and potentially better
- training
    - randomly sample our augmented data set for faster training if its taking years to compile
    - maybe double descent? Try training one with 5000+ epochs and no dropout



What I did:
1. Augmented the data to flip image 25% of the time, jitter the brightness 90% of the time, jitter the saturation 70% of the time, and jitter the contrast 60% of the time. I just thought of the differences in image quality and what would be the most likeley. *******This is subject to change throughout testing as I view this as a hyperparameter.

2. Used nn.sequential so I didn't have to pass x through a ton of stuff in the forward function. It took very little research to see that kaiming_normal_ initialization for CNN weights (especially a deep CNN like mine) is the way to go. For my two layer MLP, I went with the xavier norm, which is what I have been using to build all feedforward ML models. BatchNorm initialized to N~(0,1). 

3. At first I want to train the model straight up. Then I want to train for a long time to see if it double descends. Then I will try out pretraining
    - we see mediocre results when testing our deep CNN
    - when runnig our CNN on 400 epochs, we see some double descent, but it doesn't converge to a better test loss.
    - next I tried using a transfer learning approach that showed minimal improvement as the transfer learnng AUROC for train test and val were all 0.5 signifying that the model was no better than randomly guessing

4. At last I landed on optimizing my deep CNN and found the best val loss at epoch 37. I concatenated all of the train, test, and val data and passed it though my data augmentation wrapper and trained before testing on the challenge data.



* include ss of final_challenge_training_plot.png *


Final training should best AUROC at epoch 31. 





Challenge Overview:
After testing multiple configurations, I found that a custom-built CNN trained on all available data (train + val + test) with aggressive augmentation and regularization achieved the best performance. The final model reached a validation AUROC of 0.9986 at epoch 31, which was used for my challenge submission.

Challenge Data Augmentation:
Given the small dataset size I knew augmentation was going to be an essential step to prevent overfitting and to improve generalization. I took a probablistic approach to adding noise to data so that my model was most attuuned to normal inputs. I did horizont flips 25% of the time, and brightness, saturation and constrast jitters at 90%, 70% and 60% respectively. The intuition between probablilities of jittering the different scales was to capture what (in my mind) would be the most likely differences in images of pets. For example brightness likely changes a lot from image to image due to pet photos commonly being inside or outside. The augmentation increased the data size roughly three-fold. During training, images were augmented dynamically so the model rarely saw the same image twice.


Challenge Model Architecture:
For the architecture, I thought back to the discussion of AlexNet vs VGG from this class and my Datasci 315 class. I knew I wanted to have a deep CNN, but with limited image sizes I knew I had to think through it carefully. In order to do this, I used small filters (3x3) with stride=1 and ReLU activations throughout the model. I also used BatchNormalization and pooling in each "chunk" to add lots of parameters without running out of pixels from downsampling too much. For my MLP layer, I found that a two layer architecture performed better than a one layer. I read however that anything larger than two layers for the MLP layer is not very common or helpful. I also used nn.Sequential so that my forward block looked a little cleaner.


Challenge Regularization:
To combat overfitting, I integrated early stopping, dropout (0.5) between dense layers, and weight decay (1e-4) in the Adam optimizer. Additionally, my data augmnetation step acted as a strong regularizer. Early in development, I also ran long-duration experiments (400 epochs) to explore the double-descent phenomenon—loss briefly improved and worsened before stabilizing—but these did not outperform shorter, well-regularized training runs. After testing for double descent I was left with an unhappy laptop and possibly some dissapointment - sounds like machine learning LOL!


Challenge Hyperparameters:
The main hyperparameter tuning I played around with was dropout, learning rate (in transfer learning), and epochs. For dropout, having a low value made training much faster, but alos lead to worse test performance as expected. A dropout of 0.5 was the happy medium between having a deep CNN and limiting overfitting. As far as epochs, I tuned based on lowest val loss.


Challenge Transfer Learning Experiments:
I spent a lot of time exploring how transfer learning could help out my model, however, I became discouraged after creating 3 models and seeing train test and val AUROC scores of 0.5. Either something was horribly wrong with my code, or there wasn't much semantics to capture. If I had mre time I would'ce explored the possibility of using k means clustering as a pretraining method, but I can't image that would have been too helpful. In the end, I decided to comment out my transfer learning code tuned towards my challenge model. Maybe I bit off too much with augmentation + transfer learning, but in the end I decided to not proceed with transfer learning in my final submission.

Challenge Model Evaluation:
I evaluated each models using accuracy, loss, and AUROC on the training, validation, and test sets. I paid the most attention to the test AUROC as that is the clearest indication of how a model is generalizing to unseen data. My best validation AUROC: 0.9986 (Epoch 31) in my final model, so that's where I loaded my model from for my CSV final submission.

Overall this was a super fun and frustrating challenge, but it's always nice to come up with a model I'm somewhat proud of!
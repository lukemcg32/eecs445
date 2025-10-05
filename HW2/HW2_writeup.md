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

3. 
* attach SS of work for abc *

4. 
* attach SS of work for abc *

5. 
* attach SS of work for abc *

6. 
* attach SS of work for abc *

7.a.i.
Because df_out.ffill() only propogates forward, if a variables' first windows have no measurements, their mean will be np.nan and moving forward those missing values will remain np.nan. Also, if a patient has no readings at all for a time series variable, then every window’s mean is nan. Forward fill can’t invent a value, so the whole column for that variable remains np.nan.

7.a.ii.
Resampling creates a regular, fixed-length sequence so a standard LSTM can process the data with consistent time steps and batching. This ensures that each patient’s record has the same number of windows and variables which allows the LSTM to uniformly model changes across patients.

7.b.
weight_ih – the learnable input-hidden weights, of shape (4*hidden_size, input_size)                    - 4 * 128 * 70

weight_hh (torch.Tensor) – the learnable hidden-hidden weights, of shape (4*hidden_size, hidden_size)   - 4 * 128 * 128

bias_ih – the learnable input-hidden bias, of shape (4*hidden_size)                                     - 4 * 128

bias_hh – the learnable hidden-hidden bias, of shape (4*hidden_size)                                    - 4 * 128

total = 35840 + 65536 + 512 + 512 + (128 output weights + 1 bias from linear layer) = 102,400 + 129 = 102,529 total parameters

7.c.
* screenshot of q7_model.py *

7.d.
It looks like we start over fitting at epoch 5 when we see the validation AUROC score decline for the first time
* include screenshots of q7_auroc_og.png and q7_loss_og.png *

7.e.
=> Successfully restored checkpoint (trained for 4 epochs)
Test loss : 0.34897807240486145
Test AUROC: 0.820764163372859

7.f.i.
If Xavier initialization is replaced with constant initialization, the model will converge much more slowly or fail to converge because all neurons start identically and gradients cannot break symmetry.

7.f.ii.
Another hyperparameter we could play around with would be batch size. Lets decrease ours a little bit to 32 instead of 64 and check out performance...


7.f.iii.
You could incorporate static features by concatenating them with the LSTM’s final hidden state before the fully connected layer, so the model jointly uses time-varying and time-invariant information.
=> Successfully restored checkpoint (trained for 4 epochs)
Test loss : 0.36151470031057087
Test AUROC: 0.8145981554677207

We see a little better loss but slightly worse AUROC with the reduced batch size. Training also took much longer which was expected.

7.f.iv.
LSTMs use two states + four gates to separately control long and short-term memory, while GRUs use one state and two gates. So, replacing LSTMs with GRUs would make the model smaller and faster to train but possibly less effective at capturing very long-term data. Over the summer in my internship, I used GRUs to classify malicious vs benign LLM prompts for the AI Governance team at Lumen and it worked well!

8.a.
attention scores for the word 'ML':  [-0.3695  0.1501 -0.4157]

The largest score of 0.1501 means that the model finds the word "like" to be the most similar to "ML".

8.b.
attention weights for the word 'ML':  [0.275 0.462   0.263]
context-aware representation (context vector) for the word 'ML':  [-0.06256849  0.18003837  0.56139104]

8.c.
Multi-head attention allows the model to attend to different parts of a sentence at once which improves performance by capturing more diverse contextual relationships compared to a single attention head.

8.d.
The softmax function normalizes the attention scores into probabilities that sum to 1 and makes sure the model assigns meaningful relative importance to different words.

8.e.
Scaling by sqrt(d_k) prevents dot products from becoming too large in high-dimensional spaces and keeps the softmax function from saturating.

8.f.
Attention captures long-range dependencies by directly linking all words in a sequence. Contrary to RNNs, it doesn’t depend on step-by-step propagation.



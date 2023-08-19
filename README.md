# Lending Club

Classification code for Kaggle lending club loan data.


# Source Data

https://www.kaggle.com/datasets/deependraverma13/lending-club-loan-data-analysis-deep-learning?resource=download

# Problem

Predict loan will be paid or not.

# Split Data Program

*split\_data.py*

This program splits the data into train, test, and validation.
It also replaces periods in column names to underscores.

# Train and Predict Program

*train\_and\_predict.py*

Builds and trains a neural network to predict if the loan
was fully paid. The neural network looks like this:

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 19)]              0         
                                                                 
 dense (Dense)               (None, 20)                400       
                                                                 
 tf.nn.relu (TFOpLambda)     (None, 20)                0         
                                                                 
 dense_1 (Dense)             (None, 20)                420       
                                                                 
 tf.nn.relu_1 (TFOpLambda)   (None, 20)                0         
                                                                 
 dense_2 (Dense)             (None, 20)                420       
                                                                 
 tf.nn.relu_2 (TFOpLambda)   (None, 20)                0         
                                                                 
 dense_3 (Dense)             (None, 20)                420       
                                                                 
 tf.nn.relu_3 (TFOpLambda)   (None, 20)                0         
                                                                 
 dense_4 (Dense)             (None, 1)                 21        
                                                                 
=================================================================
Total params: 1681 (6.57 KB)
Trainable params: 1681 (6.57 KB)
Non-trainable params: 0 (0.00 Byte)

```

# Normalizing Features

The features are normalized so they will fit a neural network better.

## Categorical Features

Categorical features are converted to one-hot representation.

## Binary Features

If the feature is always 0 or 1, we leave it as-is.

## Floating Point Features

These features are transformed using
[quantile mapping.](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html)

# Balancing Training Data

Most of the data is fully paid. So training data is highly imbalanced.
In this case, we use class weights to help the neural network although
it didn't help.


# Prediction Results

This ROC curve shows true positive versus false positive.

* true positive: a bad loan was classified as bad
* false positive: a good loan was classified as bad

In this case, you would get slightly better results using the FICO
score by itself. Either way, the prediction is not very good.

<iframe src="images/roc_curve.html"></iframe>




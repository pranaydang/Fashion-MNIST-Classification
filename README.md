# Fashion-MNIST-Classification

## Explanation of Process
Approaches employed:

In order to classify the images in the Fashion MNIST dataset, I have selected 2 approaches:

1)Using a CNN whose layers are inspired by the VGG16 model

2)Generating image embeddings using the same model structure and inputting them in an SVC to classify the images

While approach number 1 is the classical approach to classify images, I was inspired by a sentiment analysis project that I took on, using BERT encodings to generate text embeddings to classify the financial news.

Metrics for success:

1)Incorporating a process for hyperparameter optimization

2)Optimize the test accuracy of the model (ideally 90%+), while taking important metrics like precision and recall into account

3)Making sure that the model does not underfit and that the the overfitting is restricted to less than 5%

Processes employed to meet metrics for success:

Some common optimization techniques are Grid Search CV and Random Search CV but the former is too computationally intensive when there are multiple hyperparameters involved and the search space is large and the latter is not deterministic. Hence, I am using Optuna which uses a method called the tree parzen estimator. Optuna keeps track of historical trials in order to guess parameters within the given ranges and tries to maximize the ratio of probability of getting a good result to the probability of getting a bad result with a combination of hyperparameters.

Optuna has an objective function which determines the metric to optimize. For the case of this model, I have tried to optimize the F1 Score, which is given by the formula: 2/((Precision)^-1 + (Recall)^-1). So, the higher the precision and recall, the higher the f1 score. In addition, the average of the f1 score can be adjusted to 'macro', which means that each f1 score of each set is weighed equally irrespective of the distribution of data, leading to a fair evaluation of the model on imbalanced data.
In the objective function of optuna, I have set a condition that heavily penalizes the model for having train accuracy - validation accuracy > 5%, effectively making sure that models from overfitting parameters are not considered.

## Approach 1: CNN

Model: I have selected the basic architecture of the VGG16 model (VGG16 Architecture). In addition, after every block (2 convolution layers & 1 max pooling layers) and after the first dense layer, I have added dropout layers that randomly remove a proportion of the neurons, making sure that the model doesn't overfit.

Preprocessing and Image Augmentation: This was very minimalistic, I only normalized the data and augmented it using the ImageDataGenerator class. I had to play around with the parameters of the ImageDataGenerator to reduce overfitting.

Optimization: The metrics that I chose to optimize were learning rate (how much the parameters change for each step), epochs (number of times a full forward and backward pass is made on the training set - avoids overfitting), batch size (number of examples per parameter update). I also tried to optimize the number of convolution blocks but decided that the number of parameters I was trying to optimize were too many, so I stuck to a fixed value.

Analysis: This model met the accuracy metric of 90%+ while keeping overfitting to less than 5% (Train accuracy: 0.9856 Test Accuracy: 0.9374 Precision: 0.9371 Recall: 0.9374). From the accuracy vs epoch and the loss vs epoch graph of the final model, we can see that there was a slight overfitting during training.

## Approach 2: Image Embeddings to SVC

Model: I am using the same CNN architecture but I am extracting the image embeddings from the flatten layer because the flattening layer effectively converts the image into a 1d array. Post this I am passing it to an SVC to classify. Having worked on text transformer architectures, I was trying to implement the concept of encoders where context (in this case, image features) are extracted to make embeddings.

Problems: The SVC takes ridiculously long to train if the train/validation ratio is 0.15 but the initial set 60000 images. So, i dropped the inital set to 10000 images and then split it into train and validation. The test set remained 10000 images. I have stratified the split to maintain the same class distribution.

Dimension reduction: In order to reduce the dimensions of the embeddings, I have used PCA to maintain a variance of 95% (since the test split generation is not deterministic, the number of dimensions corresponding to 95% variance maintained changes everytime)

Optimization: I have optimized C, kernel and gamma, which are parameters of the SVC. The SVC effectively finds the best planes to divide classes. I have kept the number of epochs for CNN fitting constant this time and have let the default values of learning rate and batch size apply.

Analysis: I was able to achieve a test accuracy of 90%+ but was not able to constrain the overfitting to less than 5%. The overfitting is still in the 5-10% range which is deemed somewhat acceptable. (Train accuracy: 0.9735 Test Accuracy: 0.9044 Precision: 0.9051 Recall: 0.9044)

## Final Evaluation

I was privileged to have a Colab Pro subsription which enabled me to use the A100 GPU. Given this system specification in mind, I believe that Approach 1, which is the CNN classification approach is better than Approach 2 using the SVC. I beleive that the SVC approach has a lot more potential and if the system allowed me to train on 60000 images, the results would be much better.

As for the scope for improvement, I believe trying out other classification methods such as Random Forest would be valuable in the Colab environment. I did not use XGBoost or LightGBM because of the large amount of time taken to train, so these options can be explored too. In addition to this, in order to generate validation sets using training, StratifiedKFold could be used as this maintains the proportion of the classes instead of random splitting.

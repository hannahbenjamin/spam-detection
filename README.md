# SMS Spam Detection using RNN

## Data Cleaning
- The dataset used is the "SMS Spam Collection" dataset available at [UCI Machine Learning Repository]([url](http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)). 
- The dataset is split into training, validation, and test sets with a stratified split to maintain the label proportions. Spam messages are augmented in the training set to address class imbalance.

## Data Preparation
- Implemented Field and Example from torchtext to process and numerically encode SMS messages.
- Moreover, constructed BucketIterator for efficient batching with minimized padding.

## Model
### Architecture
- Defined an RNN-GRU based model (SpamRNN) and an RNN model (SpamRNN2) for text classification.
- SpamRNN uses a GRU layer, while SpamRNN2 employs a basic RNN layer.
- Employed embeddings from a one-hot encoded representation of input tokens.
  
### Training and Evaluation
- The train_model function trains the RNN model with specified hyperparameters and evaluates its performance on both training and validation datasets.
- Hyperparameter tuning includes adjusting the learning rate and number of epochs, and experimenting with model variations.
- Trained models for up to 10 epochs, monitoring both training and validation accuracy and loss.
- Metrics such as accuracy and loss are plotted for analysis.

### Performance
- Evaluated different configurations: learning rate adjustments, epoch count variations, and model architectures.
- Documented the accuracy and loss metrics across different tests, with insights on model performance for each configuration.
- Achieved test accuracy of 96.5%
- The model achieved a test accuracy of 96.5%. Training metrics, including accuracy and loss, were logged and visualized for each epoch, and hyperparameter tuning was conducted to enhance performance. These steps contributed to the model's strong generalization to unseen data.

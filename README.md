# Pets-Classification
Pet breed classifier using ResNet50 transfer learning. Processes images, extracts labels, builds a model with a frozen pre-trained base, adds a classification layer, applies data augmentation, trains for 10 epochs, visualizes metrics, and evaluates on test data.

This code implements a pet breed classification model using transfer learning with ResNet50. Here's a description of what the code does:

Data Preparation:

Loads cat and dog breed images from a local directory
Extracts breed labels from filenames
Encodes 16 different breeds (6 cat breeds and 10 dog breeds) with numeric labels
Resizes all images to 224×224 pixels

Dataset Organization:

Converts images and labels to arrays
Creates one-hot encoded labels
Splits data into training (60%), validation (20%), and test (20%) sets

Model Architecture:

Implements data augmentation with horizontal flips and rotation
Uses pre-trained ResNet50 (with ImageNet weights) as the feature extractor
Freezes ResNet50 weights to prevent retraining
Adds dropout (0.2) and a final dense layer with softmax activation for 15 classes

Training:

Compiles a model with Adam optimizer and categorical cross-entropy loss
Train for 10 epochs, tracking accuracy and loss
Visualizes training/validation accuracy and loss curves

Evaluation:

Evaluate model performance on the test set
Makes predictions on test data

The model is designed for transfer learning. It uses ResNet50's pre-trained weights to classify pets into different breeds with minimal training time.

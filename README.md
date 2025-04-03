# Traffic Sign Classification

This project is part of Harvard's CS50 AI course and focuses on building a neural network to classify images of traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB). The model is built using TensorFlow and trained to identify 43 different types of road signs.

## Project Overview

The goal of this project is to:
- Preprocess and load image data from the GTSRB dataset.
- Build a Convolutional Neural Network (CNN) using TensorFlow/Keras.
- Train and evaluate the model for accuracy.
- Ensure the model achieves at least 98% accuracy on the test set.

## Dataset

The dataset used is the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html), and it can be downloaded directly from the [CS50 AI project page](https://cs50.harvard.edu/ai/2024/projects/5/traffic/).

**Note:** It is not recommended to include the dataset directly in this repository. To keep the repo lightweight and version-controlled, large files like datasets should be downloaded separately. Be sure to extract the dataset into a `data/` directory structured as follows:

## Model Architecture

The final model architecture includes:
- Convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Batch normalization for stable and faster training
- Fully connected dense layers leading to a softmax output

## Findings

During development, several experiments were conducted to improve performance:

1. **Data Augmentation Techniques**  
   Applied rotation, zoom, and horizontal flipping to training images to simulate real-world variability. This significantly improved generalization and reduced overfitting.

2. **Batch Normalization Implementation**  
   Added batch normalization layers after convolutional layers. This helped stabilize and accelerate the training process, resulting in faster convergence.

3. **Learning Rate Scheduling**  
   Used a learning rate scheduler to dynamically adjust the learning rate during training. This helped the model avoid local minima and achieve higher test accuracy.

## Results

The model achieved over **98% test accuracy**, meeting the project requirements. It was able to correctly classify most signs even in challenging conditions, demonstrating robustness and strong generalization.

## How to Run

1. Install dependencies:
   ```bash
   pip install tensorflow pillow numpy scikit-learn
   
2. Run the script:
   ```bash
   python traffic.py
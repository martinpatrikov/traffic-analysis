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
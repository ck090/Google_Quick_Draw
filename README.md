# Quickdraw-Image-Classification
Comparing different machine learning algorithms in Python to classify drawings made in the game 'Quick, draw!'.

### Results:

After training with a batch size of 10 and using 30 epochs for training both the training and validation set of images. 
The image sizes are 28x28 (std. MNIST dataset sizes)

#### The CNN is a 9-layer CNN which is composed of:
1. A convolution layer of size 5x5
2. A Max pooling layer of size 2x2
3. A smaller convolution layer with size 3x3
4. A Max pooling layer of size 2x2
5. Dropout layer with a probability of 20%
6. Flatten layer
7. Fully connected layer with 128 neurons and rectifier activation.
8. Fully connected layer with 50 neurons and rectifier activation.
9. Output layer. Keras requires one hot encoding of the y labels:


#### Classification accuracy for 2 classes (20'000 training examples):

Convolutional Neural Network: 99.9%

#### Classification accuracy for 4 classes (20'000 training examples):

Convolutional Neural Network: 99.6%

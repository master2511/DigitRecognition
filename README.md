# DigitRecognition
Using MNIST dataset to perform digit recognition using two different neural net models.
In this repository I have used two different neural network models, to recognize a given digit, i.e form 0, 1, ... 9.
I have used the MNIST dataset.
The first model is a simple Logistic Regression model.
Input layer = 28*28, output layer = 10
stochastic gradient descent is used for updating the weights, and cross entropy loss or log loss is used.

Using this simple model I m able to achieve an accuracy of 91.16.

The second model is a Multilayer neural network.
I am using 3 hidden layers.
input layer = 28*28, hidden_layer1 = 600, hidden_layer2 = 300, hidden_layer3 = 150, output layer = 10
activation function used for all the hidden layers is ReLU.
for the output layer I am using the softmax function.
BinaryCrossEntropy loss is used for loss calculation and Stochastic gradient descent is used for updating the weights.

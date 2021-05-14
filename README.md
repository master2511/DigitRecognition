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


The third model is the CNN model.
I have used two conv2d layers.
1st layer is haveing 6 filters and the kernel size is 5*5
It is followed by a max pool layer and an ReLU activation layer.
Then follows another conv2d layer of 16 filters and kernel size 5*5, which is then followed by max pool and activation layer.
I have used 3 fully connected layers in the end of sizes 120, 84, 10. Activation function used is ReLU.
Using this model i have achieved an accuracy of 97.91.

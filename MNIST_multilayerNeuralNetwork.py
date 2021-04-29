# 1 input layer 3 hidden layers which are followed by the output layer and softmax function
# cross entropy loss or log loss is used
# activation function of the hidden layers = ReLU
# stochastic gradient descent for updating the weights

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm


class MNIST_Logistic_Regression(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MNIST_Logistic_Regression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        return out


# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## Training
# Instantiate model
model = MNIST_Logistic_Regression(input_size=784, hidden_size1=600, hidden_size2=300, hidden_size3=150, num_classes=10)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
num_epoch = 2
# Iterate through train set minibatchs
for epoch in range(num_epoch):

    for images, labels in tqdm(train_loader, disable=True):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images.view(-1, 28 * 28)
        y = model(x)
        loss = criterion(y, labels)
        # Backward pass
        loss.backward()
        optimizer.step()


## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader, disable=True):
        # Forward pass
        x = images.view(-1, 28 * 28)
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct / total))
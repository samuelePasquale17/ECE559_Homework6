import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


torch.manual_seed(0)

################################################################################################
########################                     Dataset                    ########################
################################################################################################
# Download and load the MNIST training dataset
training_data = datasets.MNIST(
    root="data",  # Directory where the data will be stored
    train=True,  # Specify that this is the training set
    download=True,  # Download the dataset if not already present
    transform=ToTensor()  # Transform images to tensors for use with PyTorch
)


# Download and load the MNIST test dataset
test_data = datasets.MNIST(
    root="data",  # Directory where the data will be stored
    train=False,  # Specify that this is the test set
    download=True,  # Download the dataset if not already present
    transform=ToTensor()  # Transform images to tensors for use with PyTorch
)


# Define the batch size for loading the data
batch_size = 64


# Create data loaders to handle batching of training and test data
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# Display the shape of a sample batch of data for verification
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # N = batch size, C = channels, H = height, W = width
    print(f"Shape of y: {y.shape} {y.dtype}")  # Labels shape and data type
    break


################################################################################################
########################                  Creating model                ########################
################################################################################################
# Determine which device to use for training (CPU, GPU, or MPS)
device = (
    "cuda"
    if torch.cuda.is_available()  # Use GPU if available
    else "mps"
    if torch.backends.mps.is_available()  # Use Apple M1/M2 chip if available
    else "cpu"  # Otherwise, default to CPU
)
print(f"Using {device} device")


# Define the NeuralNetwork class inheriting from nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # Flatten the input (28x28) into a single vector (784)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 200),  # Fully connected layer from 784 inputs to 200 neurons
            nn.ReLU(),  # ReLU activation function
            nn.Linear(200, 200),  # Fully connected layer from 200 neurons to 200 neurons
            nn.ReLU(),  # ReLU activation function
            nn.Linear(200, 10),  # Fully connected layer from 200 neurons to 10 output classes
            nn.ReLU()  # ReLU activation function
        )


    def forward(self, x):
        x = self.flatten(x)  # Flatten the input image
        logits = self.linear_relu_stack(x)  # Pass the input through the network layers
        return logits


# Instantiate the model and move it to the chosen device
model = NeuralNetwork().to(device)
print(model)  # Print the model architecture


################################################################################################
########################            Optimizing model parameters         ########################
################################################################################################
# Define the loss function (CrossEntropyLoss for classification tasks)
loss_fn = nn.CrossEntropyLoss()


# Define the optimizer (Stochastic Gradient Descent with a learning rate of 0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Function to train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # Get the total number of training samples
    model.train()  # Set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Move data to the device (CPU/GPU)

        # Compute prediction and loss
        pred = model(X)  # Forward pass: get predictions
        loss = loss_fn(pred, y)  # Compute the loss

        # Backpropagation
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients

        # Print loss every 100 batches to track progress
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Function to test the model on the test dataset
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # Get the total number of test samples
    num_batches = len(dataloader)  # Get the number of batches in the test set
    model.eval()  # Set the model to evaluation mode
    test_loss, correct = 0, 0

    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Move data to the device
            pred = model(X)  # Forward pass: get predictions
            test_loss += loss_fn(pred, y).item()  # Compute total loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Count correct predictions

    test_loss /= num_batches  # Calculate average loss
    correct /= size  # Calculate accuracy
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Training loop
epochs = 5  # Number of epochs to train for
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)  # Train the model for one epoch
    test(test_dataloader, model, loss_fn)  # Evaluate the model after each epoch
print("Done!")  # Training complete
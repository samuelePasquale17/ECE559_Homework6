import torch
import torch.optim.optimizer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# seed
seed = 1317


class ConvolutionalNeuralNetwork_opt(nn.Module):
    def __init__(self, dropout_perc):
        super(ConvolutionalNeuralNetwork_opt, self).__init__()
        self.dropout_perc = dropout_perc  # Store dropout percentage

        # Convolutional block: includes two convolutional layers with ReLU activations,
        # batch normalization after each layer, a max pooling layer, and a flatten operation
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1, padding=0),  # First convolutional layer
            nn.ReLU(),  # Activation function
            nn.BatchNorm2d(20),  # Batch normalization to stabilize and optimize training
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2, padding=0),  # Second convolutional layer
            nn.ReLU(),  # Activation function
            nn.BatchNorm2d(20),  # Batch normalization
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Pooling layer to downsample feature maps
            nn.Flatten()  # Flatten feature maps into a 1D vector for input to fully connected layers
        )

        # Linear block: fully connected layers with dropout and batch normalization for regularization and stability
        self.lin_block = nn.Sequential(
            nn.Dropout(p=self.dropout_perc),  # Dropout to reduce overfitting
            nn.Linear(in_features=20 * 5 * 5, out_features=250),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.BatchNorm1d(250),  # Batch normalization
            nn.Linear(in_features=250, out_features=10),  # Output layer with 10 classes (digits 0-9)
            nn.ReLU(),  # Activation function
            nn.BatchNorm1d(10)  # Batch normalization for final output layer
        )

    def forward(self, x):
        x = self.conv_block(x)  # Pass the input through the convolutional block
        x_lin = self.lin_block(x)  # Pass the result through the linear block
        return x_lin  # Return the final output


class ConvolutionalNeuralNetwork_batch_normalization(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork_batch_normalization, self).__init__()

        # Convolutional block: consists of two convolutional layers followed by ReLU activations,
        # batch normalization, a max pooling layer, and a flatten operation to prepare for the linear layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1, padding=0),  # First convolutional layer
            nn.ReLU(),  # Activation function
            nn.BatchNorm2d(20),  # Batch normalization to stabilize training
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2, padding=0),  # Second convolutional layer
            nn.ReLU(),  # Activation function
            nn.BatchNorm2d(20),  # Batch normalization
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Pooling layer to downsample feature maps
            nn.Flatten()  # Flatten feature maps into a 1D vector
        )

        # Linear block: fully connected layers for classification, with batch normalization
        self.lin_block = nn.Sequential(
            nn.Linear(in_features=20 * 5 * 5, out_features=250),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.BatchNorm1d(250),  # Batch normalization
            nn.Linear(in_features=250, out_features=10),  # Output layer with 10 classes (digits 0-9)
            nn.ReLU(),  # Activation function
            nn.BatchNorm1d(10)  # Batch normalization
        )

    def forward(self, x):
        x = self.conv_block(x)  # Pass the input through the convolutional block
        x_lin = self.lin_block(x)  # Pass the result through the linear block
        return x_lin  # Return the final output


class ConvolutionalNeuralNetwork_dropout_regularization(nn.Module):
    def __init__(self, dropout_perc):
        super(ConvolutionalNeuralNetwork_dropout_regularization, self).__init__()
        self.dropout_perc = dropout_perc  # Store dropout percentage

        # Convolutional block: consists of two convolutional layers followed by ReLU activations,
        # a max pooling layer, and a flatten operation to prepare for the linear layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1, padding=0),  # First convolutional layer
            nn.ReLU(),  # Activation function
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2, padding=0),  # Second convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Pooling layer to downsample feature maps
            nn.Flatten()  # Flatten feature maps into a 1D vector
        )

        # Linear block: fully connected layers for classification, including dropout for regularization
        self.lin_block = nn.Sequential(
            nn.Dropout(p=self.dropout_perc),  # Dropout layer to reduce overfitting
            nn.Linear(in_features=20 * 5 * 5, out_features=250),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(in_features=250, out_features=10),  # Output layer with 10 classes (digits 0-9)
            nn.ReLU()  # Activation function
        )

    def forward(self, x):
        x = self.conv_block(x)  # Pass the input through the convolutional block
        x_lin = self.lin_block(x)  # Pass the result through the linear block
        return x_lin  # Return the final output


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # Convolutional block: consists of two convolutional layers followed by ReLU activations,
        # a max pooling layer, and a flatten operation to prepare for the linear layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1, padding=0),  # First convolutional layer
            nn.ReLU(),  # Activation function
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2, padding=0),  # Second convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Pooling layer to downsample the feature maps
            nn.Flatten()  # Flatten the feature maps into a 1D vector
        )

        # Linear block: fully connected layers to classify the features extracted by the convolutional block
        self.lin_block = nn.Sequential(
            nn.Linear(in_features=20 * 5 * 5, out_features=250),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(in_features=250, out_features=10),  # Output layer with 10 classes (digits 0-9)
            nn.ReLU()  # Activation function
        )

    def forward(self, x):
        x = self.conv_block(x)  # Pass the input through the convolutional block
        x_lin = self.lin_block(x)  # Pass the result through the linear block
        return x_lin  # Return the final output


def train(dataloader, model, loss_fn, optimizer, device):
    """
    Trains the model on the provided dataset for one epoch.

    :param dataloader: DataLoader for training data
    :param model: The neural network model to be trained
    :param loss_fn: Loss function to calculate error
    :param optimizer: Optimizer to adjust model parameters
    :param device: Device to run the model on (e.g., 'cpu' or 'cuda')
    :return: None
    """
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
            loss, current = loss.item(), (batch + 1) * len(X)  # Store loss and current sample count
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # Display progress


def test(dataloader, model, loss_fn, device):
    """
    Evaluates the model's performance on the test dataset.

    :param dataloader: DataLoader for test data
    :param model: The neural network model to evaluate
    :param loss_fn: Loss function used to calculate error
    :param device: Device to run the model on (e.g., 'cpu' or 'cuda')
    :return: None
    """
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
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")  # Output results


def get_device():
    """
    Determines the best available device for training (CPU, GPU, or MPS).

    :return: The device to be used for training
    """
    # Determine which device to use for training (CPU, GPU, or MPS)
    device = (
        "cuda"
        if torch.cuda.is_available()  # Use GPU if available
        else "mps"
        if torch.backends.mps.is_available()  # Use Apple M1/M2 chip if available
        else "cpu"  # Otherwise, default to CPU
    )
    return device  # Return the selected device


def get_dataset():
    """
    Loads the MNIST dataset for training and testing.

    :return: Tuple containing the training and test datasets
    """
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
    return training_data, test_data  # Return the training and test datasets


def init_data_loader(training_data, test_data, batch_size):
    """
    Initializes data loaders for batching the training and test datasets.

    :param training_data: Dataset used for training the model
    :param test_data: Dataset used for testing/validation
    :param batch_size: Number of samples per batch
    :return: Tuple of train and test DataLoader instances
    """
    # Create data loaders to manage batching for training and test data
    train_dataloader = DataLoader(training_data, batch_size=batch_size)  # DataLoader for training data
    test_dataloader = DataLoader(test_data, batch_size=batch_size)  # DataLoader for test data
    return train_dataloader, test_dataloader  # Return the data loaders


def init_optimization(model, learning_rate, weight_decay_enable):
    """
    Initializes the optimization setup for the model.

    :param model: The neural network model to optimize
    :param learning_rate: Learning rate for the optimizer
    :param weight_decay_enable: Boolean to enable or disable weight decay
    :return: Tuple of the loss function and optimizer
    """
    # Define the loss function (CrossEntropyLoss for classification tasks)
    # Define the optimizer (Stochastic Gradient Descent with specified learning rate)
    if weight_decay_enable:
        return nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Enable weight decay if specified
    else:
        return nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=learning_rate)  # No weight decay


def training(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs):
    """
    Trains and evaluates the model over multiple epochs.

    :param model: The neural network model to be trained
    :param train_dataloader: DataLoader for training data
    :param test_dataloader: DataLoader for testing/validation data
    :param loss_fn: Loss function to calculate the error
    :param optimizer: Optimizer to update model parameters
    :param device: Device to run the model (e.g., 'cpu' or 'cuda')
    :param epochs: Number of times to iterate over the training data
    :return: The trained model after all epochs
    """
    # Loop over each epoch
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, device)  # Train the model for one epoch
        test(test_dataloader, model, loss_fn, device)  # Evaluate the model after each epoch
    return model  # Return the trained model


def main():
    # get the device
    device = get_device()
    # get the data set
    training_data, test_data = get_dataset()
    # 50 epochs
    for epochs in [50]:
        # itarate over different batch sizes
        for batch_size in [32, 64]:
            # iterate over different learning rates
            for learning_rate in [1e-3, 1e-4, 1e-5]:
                # get data loaders
                train_dataloader, test_dataloader = init_data_loader(training_data, test_data, batch_size)

                # Model with CNN only
                torch.manual_seed(seed)
                model = ConvolutionalNeuralNetwork().to(device)
                # optimization
                loss_fn, optimizer = init_optimization(model, learning_rate, False)
                print("=====================================")
                print(f"Convolutional Neural Network:"
                      f"\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\n")
                # training
                training(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs)

                # Model with weigh_decay & dropout regularization
                for dropout_perc in [0.3, 0.4, 0.5]:
                    torch.manual_seed(seed)
                    model = ConvolutionalNeuralNetwork_dropout_regularization(dropout_perc).to(device)
                    # optimization
                    loss_fn, optimizer = init_optimization(model, learning_rate, True)
                    print("=====================================")
                    print(f"Convolutional Neural Network with weight decay and dropout regularization enabled:"
                          f"\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\n"
                          f"Dropout Percentage: {dropout_perc}")
                    # training
                    training(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs)

                # Model with batch normalization
                torch.manual_seed(seed)
                model = ConvolutionalNeuralNetwork_batch_normalization().to(device)
                # optimization
                loss_fn, optimizer = init_optimization(model, learning_rate, False)
                print("=====================================")
                print(f"Convolutional Neural Network with batch normalization enabled:"
                      f"\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\n")
                # training
                training(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs)

                # Model optimized (all features enabled)
                torch.manual_seed(seed)
                model = ConvolutionalNeuralNetwork_opt(0.5).to(device)
                # optimization
                loss_fn, optimizer = init_optimization(model, learning_rate, True)
                print("=====================================")
                print(f"Convolutional Neural Network with all features enabled:"
                      f"\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\n"
                      f"Weight decay: 1e-4\nDropout Percentage: {dropout_perc}")
                # training
                training(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs)


main()

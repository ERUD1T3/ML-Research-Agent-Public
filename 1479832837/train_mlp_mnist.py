import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Step 2: Prepare the MNIST dataset
def prepare_data_loaders(batch_size):
    # Define transformations for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Step 4: Define the multilayer perceptron (MLP) architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        all_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(all_sizes) - 1):
            self.layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

# Step 5: Initialize model, loss function, and optimizer
def initialize_model_and_optimizer(input_size, hidden_sizes, output_size, learning_rate):
    model = MLP(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

# Step 6: Implement training loop
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.view(-1, 28 * 28)  # Flatten images
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

# Step 7: Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.view(-1, 28 * 28)  # Flatten images
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Test Accuracy: {100.*correct/total:.2f}%')

# Main logic to execute the training and evaluation process
def main():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    input_size = 28 * 28
    hidden_sizes = [128, 64]
    output_size = 10

    # Prepare data
    train_loader, test_loader = prepare_data_loaders(batch_size)

    # Initialize model, loss function, and optimizer
    model, criterion, optimizer = initialize_model_and_optimizer(input_size, hidden_sizes, output_size, learning_rate)

    # Train model
    train_model(model, criterion, optimizer, train_loader, num_epochs)

        # Save the trained model
    torch.save(model.state_dict(), '1479832837/mlp_mnist.pth')
    print('Model saved to 1479832837/mlp_mnist.pth')
# Evaluate model
    evaluate_model(model, test_loader)

main()

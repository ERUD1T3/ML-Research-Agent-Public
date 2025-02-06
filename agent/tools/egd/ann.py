
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Union


class ANN(nn.Module):
    """Feed Forward Artificial Neural Network Class using PyTorch.
    
    A flexible neural network implementation that supports variable number of hidden layers.
    Inherits from PyTorch's nn.Module base class.
    
    Attributes:
        net_id (int): Unique identifier for this network
        hidden_units (List[int]): Number of units in each hidden layer
        learning_rate (float): Learning rate for optimization
        momentum (float): Momentum coefficient for optimization
        decay (float): Weight decay coefficient
        input_units (int): Number of input features
        output_units (int): Number of output units
        debug (bool): Whether to print debug information
        activation (nn.Module): Activation function to use
        output_activation (nn.Module): Output layer activations function
        topology (List[int]): Complete network architecture including input/output layers
        model (nn.Sequential): PyTorch sequential model containing all layers
        optimizer (optim.Adam): Adam optimizer for training
        device (torch.device): Device to run computations on (CPU/GPU)
    """
    def __init__(
        self, 
        net_id: int,
        hyperparams: dict,
        input_units: int, 
        output_units: int,
        debug: bool = True,
        output_activation: nn.Module = nn.Softmax(dim=-1)
    ) -> None:
        """Initialize the Artificial Neural Network.
        
        Args:
            net_id: Unique identifier for this network instance
            hyperparams: Dictionary containing network hyperparameters:
                - hidden_units: List of integers specifying hidden layer sizes
                - learning_rate: Learning rate for optimization
                - momentum: Momentum coefficient
                - decay: Weight decay coefficient
                - activation: Optional activation function (defaults to ReLU)
            input_units: Number of input features
            output_units: Number of output units/classes
            debug: Whether to print debug information
            output_activation: Activation function for output layer (defaults to Softmax)
        """
        super(ANN, self).__init__()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Print available device
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Store network hyperparameters
        self.net_id = net_id
        self.hidden_units = hyperparams['hidden_units']
        self.learning_rate = hyperparams['learning_rate']
        self.momentum = hyperparams['momentum']
        self.decay = hyperparams['decay']
        self.input_units = input_units
        self.output_units = output_units
        self.debug = debug

        # Get activation function from hyperparams or default to ReLU
        self.activation = hyperparams.get('activation', nn.ReLU())
        self.output_activation = output_activation

        # Build complete network topology: input -> hidden -> output
        self.topology = [self.input_units] + \
                        hyperparams['hidden_units'] + \
                        [self.output_units]

        # Create network layers
        layers = []
        for i in range(len(self.topology)-1):
            # Add linear layer
            layers.append(nn.Linear(self.topology[i], self.topology[i+1]))
            # Add activation after all layers
            if i < len(self.topology)-2:
                layers.append(self.activation)
            else:
                layers.append(self.output_activation)
        
        # Create sequential model from layers and move to device
        self.model = nn.Sequential(*layers).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0  # Weight decay handled manually in loss function
        )
        
    def print_weights(self) -> None:
        """Print the weights of each layer in the PyTorch neural network.
        
        Prints weight matrices for each layer, formatted with 2 decimal places.
        Each row represents the weights for a single output neuron, showing its 
        connections to all input neurons.
        
        Returns:
            None
        """
        # Iterate through named parameters of the model
        for name, param in self.model.named_parameters():
            # Only print weight matrices, skip biases
            if 'weight' in name:
                print(f'\n{name}:')
                weights = param.data
                
                # Print weights row by row
                for i in range(weights.size(0)):
                    print(f'{i}: ', end='')
                    # Print each weight in the row
                    for j in range(weights.size(1)):
                        print(f'{weights[i,j].item():.2f} ', end='')
                    print()

    def print_network(self) -> None:
        """Print the network topology and weights.
        
        Displays the complete network architecture by printing:
        1. The layer topology (number of neurons in each layer)
        2. The weight matrices between layers
        
        Returns:
            None
        """
        # Print network topology/architecture
        print('Network: ', self.topology)
        # Print weight matrices between layers
        self.print_weights()

    def set_hyperparameters(self, hyperparams: dict) -> None:
        """Set the hyperparameters and rebuild the neural network architecture.

        Updates the network's hyperparameters (learning rate, momentum, decay) and 
        rebuilds the model with a new topology based on the provided hidden units.

        Args:
            hyperparams: Dictionary containing hyperparameter values with keys:
                - 'learning_rate': Learning rate for gradient descent
                - 'momentum': Momentum coefficient for weight updates
                - 'decay': Weight decay coefficient for regularization
                - 'hidden_units': List of integers specifying number of neurons per hidden layer

        Returns:
            None
        """
        # Set training hyperparameters
        self.learning_rate = hyperparams['learning_rate']
        self.momentum = hyperparams['momentum'] 
        self.decay = hyperparams['decay']
        self.hidden_units = hyperparams['hidden_units']

        # Build full topology: input layer + hidden layers + output layer
        self.topology = [self.input_units] + \
            hyperparams['hidden_units'] + \
            [self.output_units]
        
        # Rebuild model with new topology
        layers = []
        for i in range(len(self.topology)-1):
            # Add linear layer between each pair of adjacent layers
            layers.append(nn.Linear(self.topology[i], self.topology[i+1]))
            # Add activation after all but the final layer
            if i < len(self.topology)-2:
                layers.append(self.activation)
        
        # Create new sequential model with updated architecture and move to device
        self.model = nn.Sequential(*layers).to(self.device)

        # Update optimizer with new parameters and hyperparameters
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=0  # Weight decay handled manually in loss function
        )
    
    def num_params(self) -> int:
        """Calculate total number of trainable parameters in the PyTorch model.
        
        Counts the total number of weights and biases across all layers
        in the neural network.
        
        Returns:
            int: Total number of trainable parameters in the model
        """
        # Sum up parameters across all layers using PyTorch's built-in functionality
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, filename: Union[str, None] = None) -> None:
        """Save the PyTorch model's state dictionary to a file.
        

        Saves the model's parameters (weights and biases) to a file using
        PyTorch's save functionality.

        Args:
            filename: Optional path to save the model. If not provided,
                     defaults to 'model.pt'

        Returns:
            None
        """
        # Use default filename if none provided
        filename = filename or 'model.pt'
        
        # Save model state dictionary using PyTorch's save
        torch.save(self.model.state_dict(), filename)

    def load(self, filename: str) -> None:
        """Load a saved PyTorch model's state dictionary from a file.
        
        Loads previously saved model parameters (weights and biases) from a file
        using PyTorch's load functionality.

        Args:
            filename: Path to the saved model file (.pt extension)

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file does not exist
            RuntimeError: If the loaded state dict is not compatible with current model
        """
        # Load the saved state dictionary using PyTorch's load
        state_dict = torch.load(filename, map_location=self.device)
        
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        
        # Set model to evaluation mode
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PyTorch neural network.
        
        Performs forward propagation through the network layers using PyTorch's
        built-in functionality. Each layer applies a linear transformation followed
        by the specified activation function (except the output layer).

        Args:
            x: Input tensor of shape (batch_size, input_units)
                containing the input features

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_units)
                containing the network predictions
        """
        # Flatten input if needed (e.g. for image data)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # Move input to device and forward pass through sequential model
        x = x.to(self.device)
        return self.model(x)

    def predict(self, instance: torch.Tensor) -> torch.Tensor:
        """Make predictions using the PyTorch neural network model.
        
        Performs forward pass through the network to generate predictions
        for the given input instance.

        Args:
            instance: Input tensor of shape (batch_size, input_units) or 
                     (batch_size, channels, height, width) for image data

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_units)
                         containing model predictions
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions using forward pass
        with torch.no_grad():
            predictions = self.forward(instance)
            
        return predictions

    def loss(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        no_decay: bool = False,
        loss_fn: torch.nn.Module = nn.CrossEntropyLoss()
    ) -> torch.Tensor:
        """Compute the loss for the neural network.
        
        Calculates loss between target and predicted output, with optional L2 regularization.
        Uses specified loss function (defaults to CrossEntropyLoss) and adds weight decay
        term if no_decay is False.

        Args:
            target: Target tensor of shape (batch_size, output_units)
            output: Model output tensor of shape (batch_size, output_units) 
            no_decay: If True, skip L2 regularization term
            loss_fn: Loss function to use, defaults to CrossEntropyLoss

        Returns:
            torch.Tensor: Scalar tensor containing the computed loss
        """
        # Move target to device and calculate main loss
        target = target.to(self.device)
        loss = loss_fn(output, target)

        if no_decay:
            return loss

        # Add L2 regularization term
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in self.model.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        
        # Add weight decay term scaled by number of parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        loss += self.decay * (l2_reg / (2 * n_params))

        return loss

    def training_step(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: Union[int, float, None] = 0.2) -> float:
        """Perform one training step on the given training data using minibatches.
        
        Takes training examples, splits into minibatches, performs forward and backward passes,
        and updates model parameters using SGD with momentum and weight decay.

        Args:
            train_data: List of (input, target) tuples where:
                - input is a tensor of shape (input_units,)
                - target is a tensor of shape (output_units,)
            batch_size: Size of minibatches to use. Can be:
                - None: use entire dataset as one batch. No stochasticity in gradient descent.
                - int > 0: use fixed batch size.
                - float between 0 and 1: use as percentage of dataset size.
                (default: 0.2) for 20% of dataset size.


        Returns:
            float: Average loss over all training examples

        Raises:
            ValueError: If train_data is empty or batch_size is invalid
        """
        if not train_data:
            raise ValueError('No training data provided')

        # Shuffle training data
        random.shuffle(train_data)

        # Track total loss
        total_loss = 0.0

        # Set model to training mode
        self.model.train()

        # Determine actual batch size
        n_samples = len(train_data)
        if batch_size is None:
            actual_batch_size = n_samples
        elif isinstance(batch_size, float):
            if not 0 < batch_size <= 1:
                raise ValueError('Batch size as percentage must be between 0 and 1')
            actual_batch_size = max(1, int(n_samples * batch_size))
        else:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError('Batch size must be None, float between 0-1, or positive integer')
            actual_batch_size = batch_size

        # Create batches
        n_batches = (n_samples + actual_batch_size - 1) // actual_batch_size  # Ceiling division

        # Process each batch
        for i in range(n_batches):
            start_idx = i * actual_batch_size
            end_idx = min(start_idx + actual_batch_size, n_samples)
            batch = train_data[start_idx:end_idx]

            # Stack inputs and targets into batches
            batch_inputs = torch.stack([x[0] for x in batch])
            batch_targets = torch.stack([x[1] for x in batch])

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            batch_outputs = self.forward(batch_inputs)

            # Compute loss with L2 regularization
            loss = self.loss(batch_targets, batch_outputs)
            total_loss += loss.item() * len(batch)  # Scale loss by batch size

            # Backward pass
            loss.backward()

            # Update weights using optimizer
            self.optimizer.step()

        # Return average loss
        return total_loss / n_samples

    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]], epochs: int = 10) -> float:
        """Train the neural network for multiple epochs.
        
        Performs multiple epochs of training using the provided training data.
        For each epoch, shuffles data and performs training steps.

        Args:
            train_data: List of (input, target) tuples for training
            epochs: Number of training epochs (default: 10)

        Returns:
            float: Final average loss over last epoch

        Raises:
            ValueError: If train_data is empty
        """
        if not train_data:
            raise ValueError('No training data provided')

        # Train for specified number of epochs
        for epoch in range(epochs):
            # Perform one training step on all data
            avg_loss = self.training_step(train_data)
            
            # if self.debug:
            print(f'Epoch {epoch+1} | Net #{self.net_id}\'s loss: {avg_loss:.4f}')

        return avg_loss

    def test(self, test_data: List[Tuple[torch.Tensor, torch.Tensor]], acc_report: bool = False) -> Union[float, Tuple[float, float]]:
        """Evaluate the neural network on test data.
        

        Performs forward passes on test data and computes average error and accuracy.
        Accuracy is measured as percentage of correct predictions.
        Sets model to evaluation mode during testing.

        Args:
            test_data: List of (input, target) tuples for testing
            acc_report: Whether to also return accuracy (default: False)

        Returns:
            float: Average error on test data if acc_report is False
            Tuple[float, float]: Average error and accuracy if acc_report is True

        Raises:
            ValueError: If test_data is empty
        """
        if not test_data:
            raise ValueError('No test data provided')

        # Set model to evaluation mode
        self.model.eval()
        
        total_error = 0.0
        correct = 0
        total = 0
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for inputs, targets in test_data:
                # Forward pass
                outputs = self.forward(inputs)
                
                # Compute error using MSE loss without weight decay
                error = self.loss(targets, outputs).item()
                total_error += error
                
                if acc_report:
                    # Get predicted class (max probability)
                    _, predicted = torch.max(outputs, 0)
                    # Get target class
                    _, target_class = torch.max(targets, 0)
                    # Update accuracy counts
                    total += 1
                    correct += (predicted == target_class).item()

        # Calculate average error
        avg_error = total_error / len(test_data)
        
        if acc_report:
            # Calculate accuracy as percentage correct
            accuracy = correct / total
            return avg_error, accuracy
            
        # Return just error by default
        return avg_error

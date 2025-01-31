

# imports
from ann import ANN
import random
from copy import deepcopy
import utils
import torch
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Optional

# Evolutionary Gradient Descent
class EGD:
    """Evolutionary Gradient Descent.

    Builds on top of DeepMind's PBT algorithm and augments it to also optimize 
    neural network architecture. Uses PyTorch models and supports MNIST dataset.

    Attributes:
        population_size (int): Size of population (min 20)
        population (List[ANN]): List of neural networks in population
        hyperparams (List[dict]): Hyperparameters for each network
        perfs (List[float]): Performance metrics for each network
        accuracies (List[float]): Accuracy metrics for each network
        leaderboard (List[int]): Network indices sorted by performance
        last_ready (List[int]): Last ready timestep for each network
        epochs (int): Number of training epochs
        debug (bool): Whether to print debug information
        data_percent (float): Fraction of dataset to use
        training (List[Tuple[List, List]]): Training data (inputs, targets)
        validation (List[Tuple[List, List]]): Validation data
        testing (List[Tuple[List, List]]): Test data
        n_examples (int): Total number of training examples
        input_units (int): Number of input features (784 for MNIST)
        output_units (int): Number of output classes (10 for MNIST)
    """
    
    def __init__(
        self, 
        population_size: int,
        generations: int,
        epochs: int = 500,
        debug: bool = True
    ) -> None:
        """Initialize APBT.


        Args:
            population_size: Size of population (min 20)
            epochs_num: Number of training epochs
            debug: Whether to print debug information
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Population attributes
        self.population_size = population_size
        self.population: List[ANN] = [None] * population_size
        self.hyperparams: List[dict] = [None] * population_size
        self.perfs: List[float] = [0.0] * population_size
        self.accuracies: List[float] = [0.0] * population_size
        self.leaderboard: List[int] = list(range(population_size))
        self.last_ready: List[int] = [0] * population_size
        self.generations = generations
        self.epochs = epochs
        self.debug = debug



        # Dataset configuration
        self.data_percent = 1
        self.training: List[Tuple[List, List]] = []
        self.testing: List[Tuple[List, List]] = []
        self.validation: List[Tuple[List, List]] = []
        self.n_examples = 0
        self.load_mnist_data()

        # Network architecture
        self.input_units = 28 * 28  # Flattened MNIST images
        self.output_units = 10      # MNIST classes

        # Hyperparameter ranges
        self.LR_RANGE = (1e-3, 1e-1)  # Learning rate
        self.M_RANGE = (0.0, 0.9)     # Momentum
        self.D_RANGE = (0.0, 0.1)     # Weight decay
        self.HL_RANGE = (1, 4)        # Hidden layers
        self.HUPL_RANGE = (128, 256)   # Units per layer
        self.PERTS = (0.8, 1.2)       # Perturbation factors
        
        # Training configuration
        self.READINESS = 20  # Epochs before exploitation
        self.TRUNC = 0.2      # Truncation threshold
        self.X = 1.09         # Performance scaling factor
        self.Y = 1.02        # Accuracy scaling factor

        # Initialize population
        self.generate_population(population_size)

        # Track best performers
        self.best: Optional[Tuple[ANN, float, float, dict]] = None
        self.most_acc: Optional[Tuple[ANN, float, float, dict]] = None
        self.log_path = ''

        if self.debug:
            self._print_debug_info()
            
    def _print_debug_info(self) -> None:
        """Print debug information about population and dataset."""
        print('Population:', self.population)
        print('Hyperparams:', self.hyperparams)
        print('Perfs:', self.perfs)
        print('last_ready:', self.last_ready)
        print('Training:', len(self.training))
        print('validation:', len(self.validation))
        print('Testing:', len(self.testing))
        print('Number of examples:', self.n_examples)

    def load_mnist_data(self) -> None:
        """Load and preprocess the MNIST dataset.
        
        Downloads MNIST dataset if not present, applies normalization transforms,
        converts data into PyTorch tensors, and splits into train/val/test sets.
        
        The data is:
        1. Normalized using mean 0.1307 and std 0.3081
        2. Flattened from 28x28 images to 784-dim vectors 
        3. Subsampled according to self.data_percent
        4. Split into training (80%) and validation (20%) sets
        
        Sets the following instance attributes:
            self.training: List of (input, target) tensor tuples for training
            self.validation: List of (input, target) tensor tuples for validation  
            self.testing: List of (input, target) tensor tuples for testing
            self.n_examples: Total number of training examples
        """
        # Define normalization transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Download and load MNIST datasets
        mnist_train = torchvision.datasets.MNIST(
            root='./data', 
            train=True,
            download=True,
            transform=transform
        )
        mnist_test = torchvision.datasets.MNIST(
            root='./data', 
            train=False,
            transform=transform
        )

        # Process training data using data_percent
        train_size = int(len(mnist_train) * self.data_percent)
        self.training = []
        for i in range(train_size):
            img, label = mnist_train[i]
            # Flatten image and convert to tensor
            img_flat = img.view(-1).to(self.device)
            # Convert label to one-hot tensor
            label_onehot = torch.zeros(10, device=self.device)
            label_onehot[label] = 1.0
            self.training.append((img_flat, label_onehot))

        # Split training data into train/validation sets
        self.n_examples = len(self.training)
        random.shuffle(self.training)
        split_idx = int(self.n_examples * 0.2)
        self.validation = self.training[:split_idx]
        self.training = self.training[split_idx:]

        # Process test data using data_percent
        test_size = int(len(mnist_test) * self.data_percent)
        self.testing = []
        for i in range(test_size):
            img, label = mnist_test[i]
            img_flat = img.view(-1).to(self.device)
            label_onehot = torch.zeros(10, device=self.device)
            label_onehot[label] = 1.0
            self.testing.append((img_flat, label_onehot))

    def generate_net(self, idx: int) -> Tuple[ANN, dict]:
        """Generate a new neural network with random hyperparameters.
        
        Creates a new ANN instance with randomly initialized hyperparameters within 
        predefined ranges. The network architecture and training parameters are 
        randomly sampled.

        Args:
            idx: Unique identifier for the network

        Returns:
            Tuple containing:
                - ANN: The generated neural network instance
                - dict: The randomly generated hyperparameters dictionary with keys:
                    - learning_rate: Learning rate for optimization
                    - momentum: Momentum coefficient 
                    - decay: Weight decay coefficient
                    - hidden_units: List of integers for hidden layer sizes
        """
        # Generate random hyperparameters within defined ranges
        hyperparams = {
            'learning_rate': random.uniform(*self.LR_RANGE),
            'momentum': random.uniform(*self.M_RANGE), 
            'decay': random.uniform(*self.D_RANGE),
            # Generate random architecture with variable hidden layers
            'hidden_units': [
                random.randint(*self.HUPL_RANGE)  # Units per layer
                for _ in range(random.randint(*self.HL_RANGE))  # Number of hidden layers
            ]
        }

        # Create new neural network with generated hyperparameters
        net = ANN(
            net_id=idx,
            hyperparams=hyperparams,
            input_units=self.input_units, 
            output_units=self.output_units,
            debug=self.debug
        )

        return net, hyperparams
       
    def generate_population(self, population_size: int) -> None:
        """Generate initial population of neural networks.
        
        Creates population_size neural networks with random architectures and hyperparameters.
        Stores the networks and their hyperparameters in the population and hyperparams lists.

        Args:
            population_size: Number of neural networks to generate

        Returns:
            None: Updates self.population and self.hyperparams in-place
        """
        # Generate population_size networks with random architectures/hyperparams
        for n in range(population_size):
            # Create new network and get its hyperparameters
            net, hyperparams = self.generate_net(n)
            
            # Store network and hyperparams in population
            self.population[n] = net  # Store PyTorch model
            self.hyperparams[n] = hyperparams  # Store hyperparameter dict

    def step(self, net: ANN) -> ANN:
        """Apply optimization steps to the neural network.
        
        Performs multiple training steps on the network using the current training data
        and hyperparameters. The network's optimizer and parameters are updated
        during training. Prints progress for each epoch.

        Args:
            net: Neural network instance to train

        Returns:
            ANN: The trained neural network
        """
        # Train for self.epochs
        for epoch in range(self.epochs):
            # Perform training step and get loss
            total_loss = net.training_step(self.training, batch_size=25000)
            # Print progress
            print(f'Net #{net.net_id} | Epoch {epoch+1} | Loss: {total_loss:.4f}')

        # Return the updated network
        return net

    def evaluate(self, net: ANN) -> Tuple[float, float]:
        """Evaluate the performance and accuracy of a neural network.
        
        Computes network size and validation accuracy, then calculates an overall
        performance metric using the fitness function f().

        Args:
            net: Neural network instance to evaluate

        Returns:
            Tuple[float, float]: Performance metric and accuracy, where:
                - Performance is a combined score of accuracy and network size
                - Accuracy is the validation set accuracy between 0 and 1
        """
        # Get total number of trainable parameters
        size = net.num_params()
        
        # Get error and accuracy from validation set
        _, accuracy = net.test(self.validation, acc_report=True)
        
        # Calculate performance metric using fitness function
        perf = self.f(acc=accuracy, size=size)
        
        # Return performance and accuracy
        return perf, accuracy
    
    
    def f(self, acc: float, size: int) -> float:
        """Calculate fitness score balancing accuracy and model size.
        
        Computes a fitness metric that rewards higher accuracy while penalizing
        larger model sizes. Uses exponential scaling factors X and Y to control
        the relative importance of accuracy vs size.

        Args:
            acc: Model accuracy between 0 and 1
            size: Number of trainable parameters in model

        Returns:
            float: Fitness score, higher is better
        """

        # print size and acc
        # print(f'Size: {size}, Acc: {acc}')

        # Convert accuracy to percentage (0-100) and apply reward factor X
        acc_reward = self.X ** (acc * 100)
        
        # Apply size penalty using factor Y
        size_penalty = size
        
        # Return combined fitness score
        return acc_reward / size_penalty
    

    def exploit(self, net: ANN, hyperparams: dict) -> Tuple[ANN, dict]:
        """Exploit better solutions from the population using truncation selection.
        
        Checks if the given network is in the bottom percentage of performers.
        If so, copies architecture and hyperparameters from a randomly selected 
        top performer. Otherwise keeps the current network unchanged.

        Args:
            net: Neural network to potentially replace
            hyperparams: Current hyperparameters dictionary

        Returns:
            Tuple[ANN, dict]: Either:
                - Copy of top performing network and its hyperparameters if current
                  network is in bottom percentage
                - Current network and hyperparameters unchanged if performing adequately
        """
        # Get index of current network
        index = net.net_id
        
        # Calculate bottom percentage threshold (e.g. bottom 20%)
        bottom = 1 - self.TRUNC 
        bottoms = self.leaderboard[int(self.population_size * bottom):]
        
        # Check if current network is in bottom percentage
        if index in bottoms:
            # Get indices of top performing networks
            top = self.TRUNC  # e.g. top 20%
            tops = self.leaderboard[:int(self.population_size * top)]
            
            # Randomly select one of the top performers
            top_index = random.choice(tops)
            
            # Create new network with top performer's architecture
            top_hyperparams = deepcopy(self.hyperparams[top_index])
            top_net = ANN(
                net_id=net.net_id,
                hyperparams=top_hyperparams,
                input_units=self.input_units,
                output_units=self.output_units,
                debug=self.debug
            )
            
            # Copy trained weights from top performer
            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                    top_net.model.named_parameters(),
                    self.population[top_index].model.named_parameters()
                ):
                    if param1.shape == param2.shape:
                        param1.copy_(param2)
                    

            return top_net, top_hyperparams
            
        else:
            # Network is performing adequately, keep as is
            return net, hyperparams

    def update_leaderboard(self) -> None:
        """Update the leaderboard by sorting networks based on performance.
        
        Sorts the population indices based on their performance metrics in descending order
        and updates the leaderboard attribute. The leaderboard maintains indices of networks
        ordered from best to worst performing.
        """
        # Create list of network indices sorted by performance scores
        sorted_nets = list(range(self.population_size))
        sorted_nets.sort(key=lambda x: self.perfs[x], reverse=True)
        
        # Update leaderboard with sorted indices
        self.leaderboard = sorted_nets

    def explore(self, net: ANN, hyperparams: dict) -> Tuple[ANN, dict]:
        """Explore new hyperparameters by perturbing current configuration.
        
        Randomly perturbs hyperparameters and network architecture to explore new 
        configurations. Updates both hyperparameters and network structure.

        Args:
            net: Neural network to explore
            hyperparams: Current hyperparameters dictionary

        Returns:
            Tuple containing:
                - ANN: Neural network with potentially modified architecture
                - dict: Updated hyperparameters dictionary
        """
        # Perturb learning hyperparameters
        hyperparams['learning_rate'] *= random.choice([*self.PERTS])
        hyperparams['momentum'] *= random.choice([*self.PERTS]) 
        hyperparams['decay'] *= random.choice([*self.PERTS])

        # Randomly select a hidden layer to modify
        if len(hyperparams['hidden_units']) > 0:
            layer_idx = random.randint(0, len(hyperparams['hidden_units'])-1)
            
            # Randomly add/remove/keep same number of units
            units_delta = random.choice([-1, 0, 1])
            new_units = hyperparams['hidden_units'][layer_idx] + units_delta
            
            # Ensure minimum of 1 unit per layer
            if new_units >= 1:
                hyperparams['hidden_units'][layer_idx] = new_units
                
                # Create new network with updated architecture
                new_net = ANN(
                    net_id=net.net_id,
                    hyperparams=hyperparams,
                    input_units=self.input_units,
                    output_units=self.output_units,
                    debug=self.debug
                )
                
                # Copy trained weights where possible
                with torch.no_grad():
                    # Copy weights for unchanged layers
                    for (name1, param1), (name2, param2) in zip(
                        new_net.model.named_parameters(), 
                        net.model.named_parameters()
                    ):
                        if param1.shape == param2.shape:
                            param1.copy_(param2)
                            
                return new_net, hyperparams

        # Return original if no valid architectural changes
        return net, hyperparams

    def is_ready(self, last_ready: int, timestep: int, net_id: int) -> bool:
        """Check if a network is ready for exploitation and exploration.
        
        Determines if enough epochs have passed since the network's last exploitation
        for it to be ready again. The best performing network is never ready for
        exploitation.

        Args:
            last_ready: Epoch number when network was last exploited
            timestep: Current epoch number
            net_id: ID of the network being checked

        Returns:
            bool: True if network is ready for exploitation, False otherwise
        """
        # Get current best performing network
        top_performer = self.leaderboard[0]
        
        # Best network never exploits others
        if net_id == top_performer:
            return False
            
        # Check if enough epochs have passed since last exploitation
        if timestep - last_ready > self.READINESS:
            # Update last ready timestamp
            self.last_ready[net_id] = timestep
            return True
            
        # Not enough epochs have passed
        return False
            
    def is_diff(self, net1: ANN, net2: ANN) -> bool:
        """Check if two neural networks have different parameters.
        
        Compares the parameters (weights and biases) of two neural networks
        to determine if they are different. Uses PyTorch's state_dict to
        compare parameters.

        Args:
            net1: First neural network to compare
            net2: Second neural network to compare

        Returns:
            bool: True if networks have different parameters, False otherwise
        """
        # Get state dictionaries containing model parameters
        state1 = net1.model.state_dict()
        state2 = net2.model.state_dict()

        # Compare parameters between networks
        for (name1, param1), (name2, param2) in zip(
            state1.items(),
            state2.items()
        ):
            # Check if parameter tensors are different
            if not torch.equal(param1, param2):
                return True
                
        # Networks have identical parameters
        return False

    def train(self):
        """Train the network population using evolutionary optimization.
        
        Performs evolutionary optimization over multiple epochs. For each epoch:
        1. Trains and evaluates each network in the population
        2. Updates leaderboard rankings
        3. Performs exploitation and exploration on eligible networks
        4. Tracks best performing and most accurate networks
        5. Logs training metrics
        
        Returns:
            Tuple[ANN, ANN]: Best performing network and most accurate network
        """
        # Initialize history tracking lists
        top_acc_hist = []  # Most accurate network accuracy history
        eff_acc_hist = []  # Best performing network accuracy history 
        perf_hist = []     # Best performance history
        size_hist = []     # Network size history
        lr_hist = []       # Learning rate history
        m_hist = []        # Momentum history
        d_hist = []        # Weight decay history

        # Train for specified number of epochs
        for e in range(self.generations):
            print('Generation: ', e)
            

            # Train each network in population
            for i in range(self.population_size):
                net = self.population[i]
                hyperparams = self.hyperparams[i]
                perf = self.perfs[i]
                last = self.last_ready[i]
                
                # Train and evaluate network
                net = self.step(net)
                perf, accuracy = self.evaluate(net)
                
                # Update performance metrics
                self.perfs[i] = perf
                self.accuracies[i] = accuracy
                self.update_leaderboard()

                # Check if network is ready for exploitation/exploration
                if self.is_ready(last, e, i):
                    # Try to exploit better solution
                    new_net, new_hyperparams = self.exploit(net, hyperparams)
                    
                    # If exploitation found different network, explore variations
                    if self.is_diff(new_net, net):
                        net, hyperparams = self.explore(new_net, new_hyperparams)
                        # net.set_hyperparameters(hyperparams) # not needed since explore already does this
                        

                        # Evaluate explored network
                        perf, accuracy = self.evaluate(net)
                        self.perfs[i] = perf
                        self.accuracies[i] = accuracy
                        self.update_leaderboard()

                # Update population
                self.population[i] = net
                self.hyperparams[i] = hyperparams
            
            # Update best networks
            self.best = self.get_best()
            self.most_acc = self.get_most_accurate()

            # Record history
            top_acc_hist.append(self.most_acc[2])
            eff_acc_hist.append(self.best[2])
            perf_hist.append(self.best[1])
            size_hist.append(self.best[0].num_params())
            lr_hist.append(self.best[3]['learning_rate'])
            m_hist.append(self.best[3]['momentum'])
            d_hist.append(self.best[3]['decay'])

            # Print epoch results
            print(f'Current best net perf: {self.best[1]:.2f}')
            print(f'Current best net accuracy: {self.best[2]:.2f}')
            print(f'Current best net size: {self.best[0].num_params()}')
            print(f'Current best net hyperparams: {self.best[3]}')
            print(f'Current most accurate net perf: {self.most_acc[1]:.2f}')
            print(f'Current most accurate net accuracy: {self.most_acc[2]:.2f}')
            print(f'Current most accurate net size: {self.most_acc[0].num_params()}')
            print(f'Current most accurate net hyperparams: {self.most_acc[3]}')
            
        # Log training history
        utils.log_csv(self.log_path, [
            top_acc_hist, eff_acc_hist, perf_hist, size_hist,
            lr_hist, m_hist, d_hist
        ], ['top', 'eff', 'perf', 'size', 'lr', 'm', 'd'])

        # Get final best networks
        self.best = self.get_best()
        self.most_acc = self.get_most_accurate()
        
        return self.best[0], self.most_acc[0]
        
    def get_best(self) -> Optional[Tuple[ANN, float, float, dict]]:
        """Get the neural network with best performance from population.
        
        Finds the network with the highest performance score in the current population.
        If a new best performance is found, updates and returns the new best network.
        Otherwise returns the existing best network.

        Returns:
            Optional[Tuple[ANN, float, float, dict]]: Tuple containing:
                - ANN: The best performing neural network
                - float: Performance score of the network
                - float: Accuracy score of the network
                - dict: Hyperparameters of the network
                Or None if no networks exist
        """
        # Get highest performance from current population
        best_perf = max(self.perfs)

        # Update best if none exists or new best found
        if not self.best or self.best[1] < best_perf:
            index = self.perfs.index(best_perf)
            best_net = self.population[index]
            best_hyperparams = self.hyperparams[index]
            best_acc = self.accuracies[index]
            return best_net, best_perf, best_acc, best_hyperparams
            
        # Return existing best if no improvement
        return self.best

    def get_most_accurate(self) -> Optional[Tuple[ANN, float, float, dict]]:
        """Get the neural network with highest accuracy from population.
        
        Finds the network with the highest accuracy score in the current population.
        If a new best accuracy is found, updates and returns the new most accurate network.
        Otherwise returns the existing most accurate network.

        Returns:
            Optional[Tuple[ANN, float, float, dict]]: Tuple containing:
                - ANN: The most accurate neural network
                - float: Performance score of the network
                - float: Accuracy score of the network  
                - dict: Hyperparameters of the network
                Or None if no networks exist
        """
        # Get highest accuracy from current population
        best_acc = max(self.accuracies)

        # Update most accurate if none exists or new best found
        if not self.most_acc or self.most_acc[2] < best_acc:
            index = self.accuracies.index(best_acc)
            best_net = self.population[index]
            best_hyperparams = self.hyperparams[index]
            best_perf = self.perfs[index]
            return best_net, best_perf, best_acc, best_hyperparams
            
        # Return existing most accurate if no improvement
        return self.most_acc
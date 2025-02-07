# Imports
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Tuple, Optional, Dict

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import agent.tools.egd.utils as utils
from agent.tools.egd.ann import ANN


# Evolutionary Gradient Descent
class EGD:
    """Evolutionary Gradient Descent.

    Builds on top of DeepMind's PBT algorithm and augments it to also optimize 
    neural network architecture. Uses PyTorch models and supports the MNIST dataset.

    Attributes:
        population_size (int): Size of population (min 20)
        population (Dict[int, ANN]): Dict mapping indices to neural networks
        hyperparams (Dict[int, dict]): Hyperparameters for each network
        perfs (np.ndarray): Performance metrics for each network
        accuracies (np.ndarray): Accuracy metrics for each network
        leaderboard (np.ndarray): Network indices sorted by performance
        last_ready (np.ndarray): Last ready timestep for each network
        generations (int): Number of generations for evolutionary optimization
        epochs (int): Number of training epochs per generation
        debug (bool): Whether to print debug information
        data_percent (float): Fraction of dataset to use
        training (List[Tuple[torch.Tensor, torch.Tensor]]): Training data (inputs, targets)
        validation (List[Tuple[torch.Tensor, torch.Tensor]]): Validation data
        testing (List[Tuple[torch.Tensor, torch.Tensor]]): Test data
        n_examples (int): Total number of training examples
        input_units (int): Number of input features (28*28 for MNIST)
        output_units (int): Number of output classes (10 for MNIST)
    """

    def __init__(
            self,
            population_size: int,
            generations: int,
            epochs: int = 200,
            debug: bool = True
    ) -> None:
        """Initialize EGD.

        Args:
            population_size: Size of population (min 20)
            generations: Number of generations for evolutionary training
            epochs: Number of training epochs per generation
            debug: Whether to print debug information
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Population attributes
        self.population_size = population_size
        self.population: Dict[int, ANN] = {}
        self.hyperparams: Dict[int, dict] = {}
        self.perfs = np.zeros(population_size, dtype=np.float32)
        self.accuracies = np.zeros(population_size, dtype=np.float32)
        self.leaderboard = np.arange(population_size)
        self.last_ready = np.zeros(population_size, dtype=np.int32)
        self.generations = generations
        self.epochs = epochs
        self.debug = debug

        # Dataset configuration
        self.data_percent = 1.0
        self.training: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.testing: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.validation: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.n_examples = 0
        self.load_mnist_data()

        # Network architecture
        self.input_units = 28 * 28  # Flattened MNIST images
        self.output_units = 10  # MNIST classes

        # Hyperparameter ranges
        self.LR_RANGE = (1e-3, 1e-1)  # Learning rate
        self.M_RANGE = (0.0, 0.9)  # Momentum
        self.D_RANGE = (0.0, 0.1)  # Weight decay
        self.HL_RANGE = (1, 4)  # Hidden layers count range
        self.HUPL_RANGE = (128, 256)  # Units per hidden layer range
        self.PERTS = (0.8, 1.2)  # Perturbation factors for hyperparameters

        # Training configuration
        self.READINESS = 20  # Epochs before exploitation/exploration eligibility
        self.TRUNC = 0.2  # Truncation threshold (fraction for top/bottom selection)
        self.X = 1.09  # Performance scaling factor for accuracy reward
        self.Y = 1.02  # Accuracy scaling factor (used in printing and tracking)

        # Initialize population
        self.generate_population(population_size)

        # Track best performers
        self.best: Optional[Tuple[ANN, float, float, dict]] = None
        self.most_acc: Optional[Tuple[ANN, float, float, dict]] = None
        self.log_path = ''

        # ThreadPool for concurrent GPU training
        self.executor = ThreadPoolExecutor(max_workers=self.population_size)

        if self.debug:
            self._print_debug_info()

    def _print_debug_info(self) -> None:
        """Print debug information about population and dataset."""
        print('Population:', self.population)
        print('Hyperparams:', self.hyperparams)
        print('Perfs:', self.perfs)
        print('Last ready times:', self.last_ready)
        print('Training samples:', len(self.training))
        print('Validation samples:', len(self.validation))
        print('Testing samples:', len(self.testing))
        print('Number of examples:', self.n_examples)

    def load_mnist_data(self) -> None:
        """Load and preprocess the MNIST dataset.
        
        Downloads the MNIST dataset if not present, applies normalization transforms,
        converts data into PyTorch tensors, and splits into train/val/test sets.
        
        The data is:
          1. Normalized using mean 0.1307 and std 0.3081
          2. Flattened from 28x28 images to 784-dimensional vectors 
          3. Subsampled according to self.data_percent
          4. Split into training (80%) and validation (20%) sets
        
        Sets the following instance attributes:
            self.training: List of (input, target) tensor tuples for training
            self.validation: List of (input, target) tensor tuples for validation  
            self.testing: List of (input, target) tensor tuples for testing
            self.n_examples: Total number of training examples
        """
        print("Loading MNIST data...")
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
        train_loader = torch.utils.data.DataLoader(
            mnist_train, batch_size=train_size, shuffle=True
        )
        
        # Get all training data in one batch
        images, labels = next(iter(train_loader))
        
        # Flatten images and move to device
        images = images.view(train_size, -1).to(self.device)
        
        # Convert labels to one-hot encoding
        labels_onehot = torch.zeros(train_size, 10, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1).to(self.device), 1)
        
        # Create list of tuples
        self.training = list(zip(images, labels_onehot))
        
        # Split training data into train/validation sets
        self.n_examples = len(self.training)
        split_idx = int(self.n_examples * 0.2)
        self.validation = self.training[:split_idx]
        self.training = self.training[split_idx:]

        # Process test data using data_percent
        test_size = int(len(mnist_test) * self.data_percent)
        test_loader = torch.utils.data.DataLoader(
            mnist_test, batch_size=test_size, shuffle=False
        )
        
        # Get all test data in one batch
        images, labels = next(iter(test_loader))
        
        # Flatten images and move to device
        images = images.view(test_size, -1).to(self.device)
        
        # Convert labels to one-hot encoding
        labels_onehot = torch.zeros(test_size, 10, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1).to(self.device), 1)
        
        # Create list of tuples
        self.testing = list(zip(images, labels_onehot))

        print(f'Loaded {len(self.training)} training examples, {len(self.validation)} validation examples, and {len(self.testing)} testing examples.')

    def generate_net(self, idx: int) -> Tuple[ANN, dict]:
        """Generate a new neural network with random hyperparameters.
        
        Creates a new ANN instance with randomly initialized hyperparameters within 
        the predefined ranges. The network architecture and training parameters are 
        sampled randomly.

        Args:
            idx: Unique identifier for the network

        Returns:
            Tuple containing:
              - ANN: The generated neural network instance
              - dict: The randomly generated hyperparameters dictionary with keys:
                    'learning_rate', 'momentum', 'decay', 'hidden_units'
        """
        # Generate random hyperparameters within defined ranges
        num_layers = random.randint(*self.HL_RANGE)
        hyperparams = {
            'learning_rate': random.uniform(*self.LR_RANGE),
            'momentum': random.uniform(*self.M_RANGE), 
            'decay': random.uniform(*self.D_RANGE),
            'hidden_units': np.random.randint(
                self.HUPL_RANGE[0], # lower bound
                self.HUPL_RANGE[1], # upper bound
                size=num_layers)
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
        """
        Generate the initial population of neural networks.
        
        Creates a population of neural networks with random architectures and hyperparameters.
        Stores the networks and their hyperparameters in the population and hyperparams 
        dictionaries.

        Args:
            population_size: Number of neural networks to generate

        Returns:
            None
            
        """
        # Generate all networks and hyperparams in parallel using list comprehension
        nets_and_params = [self.generate_net(n) for n in range(population_size)]
        
        # Unzip the list of tuples into separate lists
        nets, params = zip(*nets_and_params)
        
        # Update population and hyperparams dictionaries in bulk
        self.population.update({i: net for i, net in enumerate(nets)})
        self.hyperparams.update({i: param for i, param in enumerate(params)})

    async def step(self, net: ANN) -> ANN:
        """Apply optimization steps to the neural network using a thread pool.
        
        Performs multiple training steps on the network using the current training data
        and hyperparameters in a separate thread. The network's optimizer and parameters 
        are updated during training. Progress is printed for each epoch.

        Args:
            net: Neural network instance to train

        Returns:
            ANN: The trained neural network
        """

        # print freq
        print_freq = 1

        def train_net(n: ANN) -> ANN:
            """Train a neural network.
            
            Performs multiple training steps on the network using the current training data
            and hyperparameters in a separate thread. The network's optimizer and parameters 
            are updated during training. Progress is printed for each epoch.

            Args:
                n: Neural network instance to train

            Returns:
                ANN: The trained neural network
            """
            # Pre-allocate tensors and move to device
            losses = torch.zeros(self.epochs, device=n.device)
            
            # Train in batches using vectorized operations
            for epoch in range(self.epochs):
                losses[epoch] = n.training_step(self.training, batch_size=0.25)
                if epoch % print_freq == 0:
                    print(f'Net #{n.net_id} | Epoch {epoch + 1} | Loss: {losses[epoch]:.4f}')
            return n

        loop = asyncio.get_running_loop()
        trained_net = await loop.run_in_executor(self.executor, train_net, net)
        return trained_net

    def evaluate(self, net: ANN) -> Tuple[float, float]:
        """Evaluate the performance and accuracy of a neural network.
        
        Computes the network size and validation accuracy, then calculates an overall
        performance metric using a fitness function.

        Args:
            net: Neural network instance to evaluate

        Returns:
            Tuple[float, float]: Performance metric and accuracy
        """
        size = net.n_params
        _, accuracy = net.test(self.validation, acc_report=True)
        perf = self.fitness_fn(acc=accuracy, size=size)
        return perf, accuracy

    def fitness_fn(self, acc: float, size: int) -> float:
        """Calculate the fitness score balancing accuracy and model size.

        Rewards higher accuracy while penalizing larger model sizes using exponential scaling.

        Args:
            acc: Model accuracy (between 0 and 1)
            size: Number of trainable parameters in the model

        Returns:
            float: Fitness score (higher is better)
        """
        acc_reward = self.X ** (acc * 100)
        size_penalty = size
        return acc_reward / size_penalty

    def copy_net_params(self, source_net: ANN, target_net: ANN) -> None:
        """Copy matching parameters and weights between two neural networks.
        
        Efficiently copies parameters between networks, handling cases where architectures
        partially match. For mismatched layers/units, initializes with small random values.
        
        Args:
            source_net: Network to copy parameters from
            target_net: Network to copy parameters to
            
        Returns:
            None
        """
        with torch.no_grad():
            # Get state dicts for both networks
            source_state = source_net.model.state_dict()
            target_state = target_net.model.state_dict()
            
            # Track which layers were successfully copied
            copied_layers = set()
            
            # First pass - copy exact matching layers
            for target_name, target_param in target_state.items():
                if target_name in source_state:
                    source_param = source_state[target_name]
                    if target_param.shape == source_param.shape:
                        target_param.copy_(source_param)
                        copied_layers.add(target_name)
                        
            # Second pass - try partial copies for remaining layers
            for target_name, target_param in target_state.items():
                if target_name not in copied_layers:
                    if target_name in source_state:
                        source_param = source_state[target_name]
                        
                        # Handle common dimension mismatches
                        try:
                            if len(target_param.shape) == len(source_param.shape):
                                # Copy what we can
                                min_dims = [min(t, s) for t, s in zip(target_param.shape, source_param.shape)]
                                slices = tuple(slice(0, d) for d in min_dims)
                                target_param[slices].copy_(source_param[slices])
                                
                                # Initialize remaining weights
                                if target_param.shape != source_param.shape:
                                    mask = torch.ones_like(target_param, dtype=torch.bool)
                                    mask[slices] = False
                                    target_param.masked_fill_(mask, 0.0)
                                    torch.nn.init.normal_(
                                        target_param.masked_fill(~mask, 0.0), 
                                        mean=0.0, 
                                        std=0.01
                                    )

                                copied_layers.add(target_name)
                                continue
                                
                        except Exception:
                            # Fall through to re-initialization
                            # print(f'Failed to copy {target_name} from {source_net.net_id} to {target_net.net_id}')
                            pass

                    # Re-initialize if copy failed
                    torch.nn.init.normal_(target_param, mean=0.0, std=0.01)

    def exploit(self, net: ANN, hyperparams: dict) -> Tuple[ANN, dict]:
        """Exploit better solutions via truncation selection.
        
        If the given network is among the lower-performing fraction, replaces its architecture
        and hyperparameters with those of a randomly chosen top performer.

        Args:
            net: Neural network to potentially replace
            hyperparams: Current hyperparameters dictionary

        Returns:
            Tuple[ANN, dict]: Either a new network (and its hyperparameters) or the unchanged input
        """
        index = net.net_id
        bottom_threshold = 1 - self.TRUNC
        bottom_idx = int(self.population_size * bottom_threshold)
        
        # Use numpy array indexing for faster lookup
        if index in self.leaderboard[bottom_idx:]:
            # Get top performers using array slicing
            top_count = int(self.population_size * self.TRUNC)
            top_index = np.random.choice(self.leaderboard[:top_count])
            
            # Avoid deepcopy by directly accessing hyperparams
            top_hyperparams = {
                'learning_rate': self.hyperparams[top_index]['learning_rate'],
                'momentum': self.hyperparams[top_index]['momentum'],
                'decay': self.hyperparams[top_index]['decay'],
                'hidden_units': self.hyperparams[top_index]['hidden_units'].copy()
            }
            
            # Create new network
            top_net = ANN(
                net_id=net.net_id,
                hyperparams=top_hyperparams,
                input_units=self.input_units,
                output_units=self.output_units,
                debug=self.debug
            )
            
            # Copy parameters using utility function
            self.copy_net_params(self.population[top_index], top_net)
            
            return top_net, top_hyperparams
            
        return net, hyperparams

    def explore(self, net: ANN, hyperparams: dict) -> Tuple[ANN, dict]:
        """Explore new hyperparameter configurations by perturbing the current ones.
        
        Randomly perturbs learning rate, momentum, decay, and potentially the network architecture.
        If an architectural change is made, attempts to copy weights from unchanged layers.

        Args:
            net: Neural network to explore
            hyperparams: Current hyperparameters dictionary

        Returns:
            Tuple containing:
                - ANN: The neural network with updated configuration
                - dict: The updated hyperparameters dictionary
        """
        # Vectorized perturbation of hyperparameters using numpy
        perts = np.random.choice(self.PERTS, size=3)
        hyperparams = {
            'learning_rate': hyperparams['learning_rate'] * perts[0],
            'momentum': hyperparams['momentum'] * perts[1],
            'decay': hyperparams['decay'] * perts[2],
            'hidden_units': hyperparams['hidden_units'].copy()  # Preserve original array
        }

        if len(hyperparams['hidden_units']) > 0:
            # Vectorized layer modification
            layer_idx = np.random.randint(0, len(hyperparams['hidden_units']))
            units_delta = np.random.choice([-1, 0, 1])
            new_units = hyperparams['hidden_units'][layer_idx] + units_delta
            
            if new_units >= 1:
                hyperparams['hidden_units'][layer_idx] = new_units
                new_net = ANN(
                    net_id=net.net_id,
                    hyperparams=hyperparams,
                    input_units=self.input_units,
                    output_units=self.output_units,
                    debug=self.debug
                )
                
                # Copy parameters using utility function
                self.copy_net_params(net, new_net)
                            
                return new_net, hyperparams

        return net, hyperparams
    

    def update_leaderboard(self) -> None:
        """
        Update the leaderboard by sorting networks based on performance.'
        
        Sorts the networks in descending order of performance (highest first).
        """
        self.leaderboard = np.argsort(-self.perfs)

    def is_ready(self, last_ready: int, timestep: int, net_id: int) -> bool:
        """Check if a network is ready for exploitation and exploration.
        
        A network is considered ready if enough epochs have passed since its last update.
        The top performing network is never eligible.

        Args:
            last_ready: Epoch when the network was last updated
            timestep: Current epoch number
            net_id: ID of the network being checked

        Returns:
            bool: True if the network is ready, False otherwise
        """
        # Early return if network is top performer
        if net_id == self.leaderboard[0]:
            return False
            
        # Single condition check and update
        is_ready = timestep - last_ready > self.READINESS
        if is_ready:
            self.last_ready[net_id] = timestep
        return is_ready

    def is_diff(self, net1: ANN, net2: ANN) -> bool:
        """Check if two neural networks have different parameters.
        
        Compares the state dictionaries of the two networks using efficient tensor operations.

        Args:
            net1: First neural network to compare
            net2: Second neural network to compare

        Returns:
            bool: True if networks have any different parameters, False otherwise
        """
        # Get state dicts
        state1 = net1.model.state_dict()
        state2 = net2.model.state_dict()
        
        # Compare all parameters at once using torch.stack and any()
        return any(not torch.equal(state1[name], state2[name]) for name in state1.keys())

    async def train(self):
        """Train the network population using evolutionary optimization.
        
        For each generation:
          1. Trains and evaluates each network concurrently (using a thread pool).
          2. Updates leaderboard rankings.
          3. Performs exploitation and exploration on eligible networks.
          4. Tracks best performing and most accurate networks.
          5. Logs training metrics.

        Returns:
            Tuple[ANN, ANN]: Best performing network and most accurate network
        """
        # Pre-allocate history arrays
        histories = {
            'top_acc': np.zeros(self.generations),
            'eff_acc': np.zeros(self.generations), 
            'perf': np.zeros(self.generations),
            'size': np.zeros(self.generations),
            'lr': np.zeros(self.generations),
            'm': np.zeros(self.generations),
            'd': np.zeros(self.generations)
        }

        for gen in range(self.generations):
            print('Generation:', gen)

            # Train all networks in parallel
            trained_nets = await asyncio.gather(*map(self.step, self.population.values()))

            # Update population metrics using vectorized operations
            for net_id, net in enumerate(trained_nets):
                self.population[net_id] = net
                self.perfs[net_id], self.accuracies[net_id] = self.evaluate(net)

            # Update rankings
            self.update_leaderboard()

            # Exploitation and exploration
            ready_mask = np.array([
                self.is_ready(self.last_ready[net_id], gen, net_id)
                for net_id in range(self.population_size)
            ])
            ready_ids = np.where(ready_mask)[0]

            for net_id in ready_ids:
                net = self.population[net_id]
                hyperparams = self.hyperparams[net_id]
                
                new_net, new_hyperparams = self.exploit(net, hyperparams)
                if self.is_diff(new_net, net):
                    net, hyperparams = self.explore(new_net, new_hyperparams)
                    self.perfs[net_id], self.accuracies[net_id] = self.evaluate(net)
                    
                self.population[net_id] = net
                self.hyperparams[net_id] = hyperparams

            self.update_leaderboard()
            self.best = self.get_best()
            self.most_acc = self.get_most_accurate()

            # Update histories efficiently
            histories['top_acc'][gen] = self.most_acc[2]
            histories['eff_acc'][gen] = self.best[2]
            histories['perf'][gen] = self.best[1]
            histories['size'][gen] = self.best[0].n_params
            histories['lr'][gen] = self.best[3]['learning_rate']
            histories['m'][gen] = self.best[3]['momentum']
            histories['d'][gen] = self.best[3]['decay']

            # Print status
            print(f'Current best net perf: {self.best[1]:.2f}')
            print(f'Current best net accuracy: {self.best[2]:.2f}')
            print(f'Current best net size: {self.best[0].num_params()}')
            print(f'Current best net hyperparams: {self.best[3]}')
            print(f'Current most accurate net perf: {self.most_acc[1]:.2f}')
            print(f'Current most accurate net accuracy: {self.most_acc[2]:.2f}')
            print(f'Current most accurate net size: {self.most_acc[0].num_params()}')
            print(f'Current most accurate net hyperparams: {self.most_acc[3]}')

        # Log histories
        utils.log_csv(
            self.log_path, 
            [
                histories['top_acc'], 
                histories['eff_acc'], 
                histories['perf'],
                histories['size'], 
                histories['lr'], 
                histories['m'],
                histories['d']
            ], 
            ['top', 'eff', 'perf', 'size', 'lr', 'm', 'd'],
            generations=self.generations,
            epochs=self.epochs,
            plot=True
        )

        self.best = self.get_best()
        self.most_acc = self.get_most_accurate()

        return self.best[0], self.most_acc[0]

    def get_best(self) -> Optional[Tuple[ANN, float, float, dict]]:
        """Get the neural network with the best performance from the population.
        
        Returns:
            Optional[Tuple[ANN, float, float, dict]]: Tuple containing the best network, 
            its performance, accuracy, and hyperparameters, or None if no networks exist.
        """
        # Since self.perfs is already a numpy array, we can use numpy operations directly
        best_perf = np.max(self.perfs)
        
        if not self.best or self.best[1] < best_perf:
            # Get index of best performance using numpy
            index = np.argmax(self.perfs)
            # Return tuple of values at that index
            return (
                self.population[index], 
                best_perf, 
                self.accuracies[index], 
                self.hyperparams[index]
            )
            
        return self.best

    def get_most_accurate(self) -> Optional[Tuple[ANN, float, float, dict]]:
        """Get the neural network with the highest accuracy from the population.
        
        Returns:
            Optional[Tuple[ANN, float, float, dict]]: Tuple containing the most accurate network, 
            its performance, accuracy, and hyperparameters, or None if no networks exist.
        """
        # Use numpy max which is faster than Python max
        best_acc = np.max(self.accuracies)
        
        if not self.most_acc or self.most_acc[2] < best_acc:
            # Get index of max accuracy using numpy
            index = np.argmax(self.accuracies)
            # Return tuple directly using index lookups
            return (
                self.population[index], 
                self.perfs[index],
                best_acc,
                self.hyperparams[index]
            )
            
        return self.most_acc

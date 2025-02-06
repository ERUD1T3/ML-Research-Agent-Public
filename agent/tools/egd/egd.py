# Imports
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Tuple, Optional

import torch
import torchvision
import torchvision.transforms as transforms

import agent.tools.egd.utils as utils
from agent.tools.egd.ann import ANN


# Evolutionary Gradient Descent
class EGD:
    """Evolutionary Gradient Descent.

    Builds on top of DeepMind's PBT algorithm and augments it to also optimize 
    neural network architecture. Uses PyTorch models and supports the MNIST dataset.

    Attributes:
        population_size (int): Size of population (min 20)
        population (List[ANN]): List of neural networks in population
        hyperparams (List[dict]): Hyperparameters for each network
        perfs (List[float]): Performance metrics for each network
        accuracies (List[float]): Accuracy metrics for each network
        leaderboard (List[int]): Network indices sorted by performance
        last_ready (List[int]): Last ready timestep for each network
        generations (int): Number of generations for evolutionary optimization
        epochs (int): Number of training epochs per generation
        debug (bool): Whether to print debug information
        data_percent (float): Fraction of dataset to use
        training (List[Tuple[List, List]]): Training data (inputs, targets)
        validation (List[Tuple[List, List]]): Validation data
        testing (List[Tuple[List, List]]): Test data
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
        self.data_percent = 1.0
        self.training: List[Tuple[List, List]] = []
        self.testing: List[Tuple[List, List]] = []
        self.validation: List[Tuple[List, List]] = []
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
        self.training = []
        for i in range(train_size):
            img, label = mnist_train[i]
            # Flatten image and move to device
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
        hyperparams = {
            'learning_rate': random.uniform(*self.LR_RANGE),
            'momentum': random.uniform(*self.M_RANGE),
            'decay': random.uniform(*self.D_RANGE),
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
        """Generate the initial population of neural networks.
        
        Creates 'population_size' neural networks with random architectures and hyperparameters.
        Stores the networks and their hyperparameters in the population and hyperparams lists.
        """
        for n in range(population_size):
            net, hyperparams = self.generate_net(n)
            self.population[n] = net
            self.hyperparams[n] = hyperparams

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

        def train_net(n: ANN) -> ANN:
            for epoch in range(self.epochs):
                total_loss = n.training_step(self.training, batch_size=0.25)
                print(f'Net #{n.net_id} | Epoch {epoch + 1} | Loss: {total_loss:.4f}')
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
        size = net.num_params()
        _, accuracy = net.test(self.validation, acc_report=True)
        perf = self.f(acc=accuracy, size=size)
        return perf, accuracy

    def f(self, acc: float, size: int) -> float:
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
        bottoms = self.leaderboard[int(self.population_size * bottom_threshold):]

        if index in bottoms:
            top_count = int(self.population_size * self.TRUNC)
            tops = self.leaderboard[:top_count]
            top_index = random.choice(tops)
            top_hyperparams = deepcopy(self.hyperparams[top_index])
            top_net = ANN(
                net_id=net.net_id,
                hyperparams=top_hyperparams,
                input_units=self.input_units,
                output_units=self.output_units,
                debug=self.debug
            )
            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                        top_net.model.named_parameters(),
                        self.population[top_index].model.named_parameters()
                ):
                    if param1.shape == param2.shape:
                        param1.copy_(param2)
            return top_net, top_hyperparams
        else:
            return net, hyperparams

    def update_leaderboard(self) -> None:
        """Update the leaderboard by sorting networks based on performance."""
        sorted_nets = list(range(self.population_size))
        sorted_nets.sort(key=lambda x: self.perfs[x], reverse=True)
        self.leaderboard = sorted_nets

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
        hyperparams['learning_rate'] *= random.choice(self.PERTS)
        hyperparams['momentum'] *= random.choice(self.PERTS)
        hyperparams['decay'] *= random.choice(self.PERTS)

        if len(hyperparams['hidden_units']) > 0:
            layer_idx = random.randint(0, len(hyperparams['hidden_units']) - 1)
            units_delta = random.choice([-1, 0, 1])
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
                with torch.no_grad():
                    for (name1, param1), (name2, param2) in zip(
                            new_net.model.named_parameters(),
                            net.model.named_parameters()
                    ):
                        if param1.shape == param2.shape:
                            param1.copy_(param2)
                return new_net, hyperparams

        return net, hyperparams

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
        top_performer = self.leaderboard[0]
        if net_id == top_performer:
            return False
        if timestep - last_ready > self.READINESS:
            self.last_ready[net_id] = timestep
            return True
        return False

    def is_diff(self, net1: ANN, net2: ANN) -> bool:
        """Check if two neural networks have different parameters.
        
        Compares the state dictionaries of the two networks.

        Args:
            net1: First neural network to compare
            net2: Second neural network to compare

        Returns:
            bool: True if networks have any different parameters, False otherwise
        """
        state1 = net1.model.state_dict()
        state2 = net2.model.state_dict()
        for (name1, param1), (name2, param2) in zip(state1.items(), state2.items()):
            if not torch.equal(param1, param2):
                return True
        return False

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
        top_acc_hist = []
        eff_acc_hist = []
        perf_hist = []
        size_hist = []
        lr_hist = []
        m_hist = []
        d_hist = []

        for e in range(self.generations):
            print('Generation:', e)

            # Train all networks in parallel (thread pool tasks)
            tasks = [self.step(net) for net in self.population]
            trained_nets = await asyncio.gather(*tasks)

            # Update population with trained networks
            for i, net in enumerate(trained_nets):
                self.population[i] = net
                perf, accuracy = self.evaluate(net)
                self.perfs[i] = perf
                self.accuracies[i] = accuracy

            # Update rankings and perform exploitation/exploration
            self.update_leaderboard()
            for i in range(self.population_size):
                net = self.population[i]
                hyperparams = self.hyperparams[i]
                last = self.last_ready[i]

                if self.is_ready(last, e, i):
                    new_net, new_hyperparams = self.exploit(net, hyperparams)
                    if self.is_diff(new_net, net):
                        net, hyperparams = self.explore(new_net, new_hyperparams)
                        perf, accuracy = self.evaluate(net)
                        self.perfs[i] = perf
                        self.accuracies[i] = accuracy

                self.population[i] = net
                self.hyperparams[i] = hyperparams

            self.update_leaderboard()
            self.best = self.get_best()
            self.most_acc = self.get_most_accurate()

            top_acc_hist.append(self.most_acc[2])
            eff_acc_hist.append(self.best[2])
            perf_hist.append(self.best[1])
            size_hist.append(self.best[0].num_params())
            lr_hist.append(self.best[3]['learning_rate'])
            m_hist.append(self.best[3]['momentum'])
            d_hist.append(self.best[3]['decay'])

            print(f'Current best net perf: {self.best[1]:.2f}')
            print(f'Current best net accuracy: {self.best[2]:.2f}')
            print(f'Current best net size: {self.best[0].num_params()}')
            print(f'Current best net hyperparams: {self.best[3]}')
            print(f'Current most accurate net perf: {self.most_acc[1]:.2f}')
            print(f'Current most accurate net accuracy: {self.most_acc[2]:.2f}')
            print(f'Current most accurate net size: {self.most_acc[0].num_params()}')
            print(f'Current most accurate net hyperparams: {self.most_acc[3]}')

        utils.log_csv(self.log_path, [
            top_acc_hist, eff_acc_hist, perf_hist, size_hist,
            lr_hist, m_hist, d_hist
        ], ['top', 'eff', 'perf', 'size', 'lr', 'm', 'd'])

        self.best = self.get_best()
        self.most_acc = self.get_most_accurate()

        return self.best[0], self.most_acc[0]

    def get_best(self) -> Optional[Tuple[ANN, float, float, dict]]:
        """Get the neural network with the best performance from the population.
        
        Returns:
            Optional[Tuple[ANN, float, float, dict]]: Tuple containing the best network, 
            its performance, accuracy, and hyperparameters, or None if no networks exist.
        """
        best_perf = max(self.perfs)
        if not self.best or self.best[1] < best_perf:
            index = self.perfs.index(best_perf)
            best_net = self.population[index]
            best_hyperparams = self.hyperparams[index]
            best_acc = self.accuracies[index]
            return best_net, best_perf, best_acc, best_hyperparams
        return self.best

    def get_most_accurate(self) -> Optional[Tuple[ANN, float, float, dict]]:
        """Get the neural network with the highest accuracy from the population.
        
        Returns:
            Optional[Tuple[ANN, float, float, dict]]: Tuple containing the most accurate network, 
            its performance, accuracy, and hyperparameters, or None if no networks exist.
        """
        best_acc = max(self.accuracies)
        if not self.most_acc or self.most_acc[2] < best_acc:
            index = self.accuracies.index(best_acc)
            best_net = self.population[index]
            best_hyperparams = self.hyperparams[index]
            best_perf = self.perfs[index]
            return best_net, best_perf, best_acc, best_hyperparams
        return self.most_acc



# imports
from ann import ANN
import random
from copy import deepcopy
import utils
import torchvision
import torchvision.transforms as transforms

# Augmentated Population Based Training
class APBT:
    '''
    Augmentated Population Based Training
    Builts on top of Deepmind PBT algorithm
    and augments it to also optimize the 
    Neural network arcthiecture
    '''
    
    def __init__(
        self, 
        population_size,
        epochs_num,
        debug=True):
        '''
        Initialize the APBT class
        '''
        self.population_size = population_size # min = 20
        self.population = [None for _ in range(population_size)]
        self.hyperparams = [None for _ in range(population_size)]
        self.perfs = [0.0 for _ in range(population_size)]
        self.accuracies = [0.0 for _ in range(population_size)]
        self.leaderboard = [i for i in range(population_size)] # based on performance
        self.last_ready = [0 for _ in range(population_size)]
        self.epochs = epochs_num
        self.debug = debug

        # Dataset size percentage to use
        self.data_percent = 0.01

        # Load and process MNIST data
        self.training = []
        self.testing = []
        self.validation = []
        self.n_examples = 0
        self.load_mnist_data()

        # Set input/output dimensions for MNIST
        self.input_units = 28 * 28  # Flattened 28x28 images
        self.output_units = 10      # 10 digit classes
        
        # initial ranges for the constants
        self.LR_RANGE = (1e-4, 1e-1) # learning rate
        self.M_RANGE = (.0, .9) # momentum
        self.D_RANGE = (.0, .1) # decay
        self.HL_RANGE = (1, 4) # hidden layers
        self.HUPL_RANGE = (16, 256) # hidden units per layer
        self.PERTS = (0.8, 1.2) # perturbations
        self.READINESS = 220 # number of epochs to wait before exploitation
        self.TRUNC = .2 # truncation threshold
        self.X, self.Y = 1.09, 1.02 # scaling factor
    
        # generate the population
        self.generate_population(population_size)

        # best running best performer, its performance,
        # accuracy, and its hyperparameters 
        self.best = None
        self.most_acc = None
        self.log_path = ''

        if self.debug:
            print('Population:', self.population)
            print('Hyperparams:', self.hyperparams)
            print('Perfs:', self.perfs)
            print('last_ready:', self.last_ready)
            print('Training:', len(self.training))
            print('validation:', len(self.validation))
            print('Testing:', len(self.testing))
            print('Number of examples:', self.n_examples)

    def load_mnist_data(self) -> None:
        '''
        Load and preprocess the MNIST dataset.
        
        This function:
        1. Downloads MNIST if not present
        2. Applies normalization transforms
        3. Flattens images and converts labels to one-hot encoding
        4. Splits data into training/validation/test sets
        5. Subsamples data according to self.data_percent
        '''
        # Define normalization transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Download and load MNIST
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

        # Process training data using only data_percent
        train_size = int(len(mnist_train) * self.data_percent)
        for i in range(train_size):
            img, label = mnist_train[i]
            # Flatten 28x28 image to 1D array
            img_flat = img.view(-1).tolist()
            # Convert label to one-hot encoding
            label_onehot = [1.0 if i == label else 0.0 for i in range(10)]
            self.training.append([img_flat, label_onehot])

        # Split training data into train/validation sets
        self.n_examples = len(self.training)
        random.shuffle(self.training)
        split_idx = int(self.n_examples * 0.2)
        self.validation = self.training[:split_idx]
        self.training = self.training[split_idx:]

        # Process test data using only data_percent
        test_size = int(len(mnist_test) * self.data_percent)
        for i in range(test_size):
            img, label = mnist_test[i]
            img_flat = img.view(-1).tolist()
            label_onehot = [1.0 if i == label else 0.0 for i in range(10)]
            self.testing.append([img_flat, label_onehot])



    def generate_net(self, idx):
        '''
        Generate a new network
        '''
        h = {
            # 'k_fold': random.randint(2, 10),
            'learning_rate': random.uniform(*self.LR_RANGE),
            'momentum': random.uniform(*self.M_RANGE),
            'decay': random.uniform(*self.D_RANGE),
            'hidden_units': [
                random.randint(*self.HUPL_RANGE) 
                for _ in range(random.randint(*self.HL_RANGE))
            ] # list of number of nodes in each layer
        }

        net = ANN(
            net_id=idx,
            hyperparams=h, 
            input_units=self.input_units,
            output_units=self.output_units,
            debug=self.debug
        )

        return net, h
       
    def generate_population(self, population_size):
        '''
        Generate the population of neural networks
        '''
        for n in range(population_size):
            net, h = self.generate_net(n)
            self.population[n] = net
            self.hyperparams[n] = h

    def step(self, net):
        '''
        Apply one optimization step to the network,
        given the hyperparameters
        '''
        # set through training data with 
        # the hyperparameters setd
        net.training_step(self.training)
        # return the network address
        return net

    def evaluate(self, net):
        '''
        Evaluate the performance of the network
        '''
        size = net.num_params()
        accuracy = net.test(self.validation)
        perf = self.f(acc=accuracy, size=size)
        # print(f' | perf: {perf:.3f} | size: {size} | accuracy: {accuracy:.3f}', end='\n')
        return perf, accuracy
    # try to figure out what f should be
    def f(self, acc, size):
        '''
        Fitness function
        '''
        # reward for accuracy, penalty for size
        return self.X ** (acc * 100) / self.Y ** size

    def exploit(self, net, hyperparams):
        '''
        Exploit the rest of the population 
        to find a better solution
        truncation selection
        '''
        # get index of net
        index = net.net_id
        # get the bottoms
        bottom = 1 - self.TRUNC # bottom 20%
        bottoms = self.leaderboard[int(self.population_size * bottom):]
        # check if net is in the bottom 20%
        if index in bottoms:
            # get the tops
            top = self.TRUNC # top 20%
            tops = self.leaderboard[:int(self.population_size * top)]
            # get the index of one of the top 20%
            top_index = random.choice(tops)
            # get the index of the top net
            top_net = deepcopy(self.population[top_index])
            # get the hyperparameters of the top net
            top_hyperparams = deepcopy(self.hyperparams[top_index])
            # replace the current net with the top net
            return top_net, top_hyperparams
        else :
            # net is not in the bottom 20%
            # so it's doing okay for now
            return net, hyperparams

    def update_leaderboard(self):
        '''
        Update the leaderboard
        '''
        # sort the population by perfs
        sorted_nets = [i for i in range(self.population_size)]
        sorted_nets.sort(key=lambda x: self.perfs[x], reverse=True)
        # update leaderboard
        self.leaderboard = sorted_nets

    def explore(self, net, hyperparams):
        '''
        Produce new hyperparameters to explore by 
        perturbing the current hyperparameters
        '''
        # randomly perturb the hyperparameters by factor
        hyperparams['learning_rate'] *= random.choice([*self.PERTS])
        hyperparams['momentum'] *= random.choice([*self.PERTS])
        hyperparams['decay'] *= random.choice([*self.PERTS])

        # randomly perturb the topology
        rng_index = random.randint(1, len(net.topology) - 2)
        # randomly add or remove a unit
        rng_choice = random.choice([-1, 0, 1])
        # udpated the hyperparameter
        hyperparams['hidden_units'][rng_index - 1] += rng_choice
        # check if the hyperparameter is valid
        if hyperparams['hidden_units'][rng_index - 1] < 1:
            # if not, revert back to the previous hyperparameter
            hyperparams['hidden_units'][rng_index - 1] = 1
            # and return the previous hyperparameter
            return net, hyperparams 

        # adjust the weights based on changed topology
        if rng_choice == -1:
            # remove weight associated with removed unit
            # choose a random unit to remove
            rng_unit = random.randint(0, net.topology[rng_index] - 1)
            # row weight
            del net.weights[f'W{rng_index}{rng_index-1}'][rng_unit]
            # column weight
            for r in range(net.topology[rng_index+1]):
                del net.weights[f'W{rng_index+1}{rng_index}'][r][rng_unit]
            # remove the unit
            net.topology[rng_index] -= 1

        elif rng_choice == 1: # rng_choice = 1
            # row weight
            net.weights[f'W{rng_index}{rng_index-1}'].append([
                net.rand_init() for _ in range(1+net.topology[rng_index-1])])
            # column weight
            for r in range(net.topology[rng_index+1]):
                net.weights[f'W{rng_index+1}{rng_index}'][r].append(net.rand_init())
            # add the unit
            net.topology[rng_index] += 1   
        else: # rng_choice = 0
            # do nothing
            pass         

        return net, hyperparams

    def is_ready(self, last_ready, timestep, net_id):
        '''
        Check if the net is ready to exploit and explore
        after a certain number of last_ready since last ready
        '''
        # get top 3 of leaderboard
        top = self.leaderboard[0]
        # check if perf is top
        if net_id == top:
            return False # top never exploit
        # checking the readiness
        if timestep - last_ready > self.READINESS:
            self.last_ready[net_id] = timestep
            # might need to check if the performance is good enough
            return True
        # by default not ready
        return False
            
    def is_diff(sel, net1, net2):
        '''
        Check if the networks are different,
        by checking if the weights are different
        if a net is doing okay, it's not different
        '''
        # check if the weights dicts are different
        if net1.weights != net2.weights:
            return True
        # by default, they are not different
        return False

    def train(self):
        '''
        Train the network population
        '''
        top_acc_hist = []
        eff_acc_hist = []
        perf_hist = []
        size_hist = []

        lr_hist = []
        m_hist = []
        d_hist = []

        for e in range(self.epochs):
            # print the epoch number
            print('Epoch: ', e, end='\n')
            for i in range(self.population_size):
                # getting a net of the population
                net = self.population[i]
                hyperparams = self.hyperparams[i]
                perf = self.perfs[i]
                last = self.last_ready[i]
                # optimize the net
                net = self.step(net)
                # evaluate the net
                perf, accuracy = self.evaluate(net)
                # update
                self.perfs[i] = perf
                self.accuracies[i] = accuracy
                # update the leaderboard
                self.update_leaderboard()

                # check if the net is ready to exploit and explore
                if self.is_ready(last, e, i):
                    new_net, new_hyperparams = self.exploit(net, hyperparams)
                    # check if the new network is different
                    if self.is_diff(new_net, net): 
                        # have you copied the best
                        net, hyperparams = self.explore(new_net, new_hyperparams)
                        # set the hyperparameters
                        net.set_hyperparameters(hyperparams)
                        # evaluate the net with perturbations
                        perf, accuracy = self.evaluate(net)
                        # update
                        self.perfs[i] = perf
                        self.accuracies[i] = accuracy
                        # update the leaderboard
                        self.update_leaderboard()

                # update the population
                self.population[i] = net
                self.hyperparams[i] = hyperparams
            
            # get the most accurate net so far 
            self.best = self.get_best()
            self.most_acc = self.get_most_accurate()

            # update the histories
            top_acc_hist.append(self.most_acc[2])
            eff_acc_hist.append(self.best[2])
            perf_hist.append(self.best[1])
            size_hist.append(self.best[0].num_params())
            lr_hist.append(self.best[3]['learning_rate'])
            m_hist.append(self.best[3]['momentum'])
            d_hist.append(self.best[3]['decay'])

            # print the best net so far
            print(f'Current best net perf: {self.best[1]:.2f}', end='\n')
            print(f'Current best net accuracy: {self.best[2]:.2f}', end='\n')
            print(f'Current best net size: {self.best[0].num_params()}', end='\n')
            print(f'Current best net hyperparams: {self.best[3]}', end='\n')
            # print the most accurate net so far
            print(f'Current most accurate net perf: {self.most_acc[1]:.2f}', end='\n')
            print(f'Current most accurate net accuracy: {self.most_acc[2]:.2f}', end='\n')
            print(f'Current most accurate net size: {self.most_acc[0].num_params()}', end='\n')
            print(f'Current most accurate net hyperparams: {self.most_acc[3]}', end='\n')
            
        # log the histories
        utils.log_csv(self.log_path, [
            top_acc_hist, eff_acc_hist, perf_hist, size_hist,
            lr_hist, m_hist, d_hist
        ], ['top', 'eff', 'perf', 'size', 'lr', 'm', 'd'])
        # get most accurate overall
        self.best = self.get_best() # might not be necessary
        self.most_acc = self.get_most_accurate()
        # return the best net
        return self.best[0], self.most_acc[0]
        
    def get_best(self):
        '''
        Get the best net
        '''
        # max last gen perf
        best_perf = max(self.perfs)
        if not self.best or self.best[1] < best_perf: # if no best net yet or new best net found
            index = self.perfs.index(best_perf)
            best_net = self.population[index]
            best_hyperparams = self.hyperparams[index]
            best_acc = self.accuracies[index]
            return best_net, best_perf, best_acc, best_hyperparams
        else: # best net is the same
            return self.best

    def get_most_accurate(self):
        '''
        Get the most accurate net
        '''
        # max last gen accuracy
        best_acc = max(self.accuracies)
        if not self.most_acc or self.most_acc[2] < best_acc: # if no best net yet or new best net found
            index = self.accuracies.index(best_acc)
            best_net = self.population[index]
            best_hyperparams = self.hyperparams[index]
            best_perf = self.perfs[index]
            return best_net, best_perf, best_acc, best_hyperparams
        else: # best net is the same
            return self.most_acc
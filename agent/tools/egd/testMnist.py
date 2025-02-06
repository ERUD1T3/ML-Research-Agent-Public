# imports
from agent.tools.egd.egd import EGD
import asyncio

# TODO: 
# 0. make the egd steps faster, for faster generations ops!!!
# 1. add a more to increase or decrease the delta in hyperparams changes. a sort of acceleration factor to explore faster.
# 2. add more layer building blocks like BatchNorm, Dropout, etc.
# 3. add support for AdamW optimizer
# 4. add ability to mix architectures like convs, fcs, attention, etc.
# 5. add ability to train on multiple GPUs
# 6. add ability to train on multiple nodes




def main():
    '''main of the program'''

    debugging = True
    population_size = 30  # population size, 80 good number
    generations = 20  # number of epochs
    epochs = 150

    # run trials
    for trial in range(1):
        egd = EGD(population_size, generations, epochs, debugging)
        # set log path
        egd.log_path = f'logs/mnist{trial}.csv'

        print('Trial: ', trial)
        print('\nRunning the population based training\n')
        # asyncio.run() executes the coroutine egd.train() in an event loop,
        # allowing concurrent training of multiple networks in the population
        best_net, most_acc = asyncio.run(egd.train())
        print('\nPopulation Based Training complete\n')
        # create the artificial neural network

        # printing the neural network
        print('\nPrinting learned weights of best\n')
        best_net.print_network()
        # test the artificial neural network
        print('\nTesting the NN...\n')
        accuracy = 100 * best_net.test(egd.testing)
        n_params = best_net.num_params()
        print('\nTesting complete\n')
        print(f'\nAccuracy: {accuracy:.2f}%\n')

        print(f'Number of parameters: {n_params}\n')

        # printing the neural network
        print('\nPrinting learned weights of the most accurate\n')
        most_acc.print_network()
        # test the artificial neural network
        print('\nTesting the NN...\n')
        accuracy = 100 * most_acc.test(egd.testing)
        n_params = most_acc.num_params()
        print('\nTesting complete\n')
        print(f'\nAccuracy: {accuracy:.2f}%\n')
        print(f'Number of parameters: {n_params}\n')


if __name__ == '__main__':
    main()

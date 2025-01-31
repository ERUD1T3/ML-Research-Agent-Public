# imports
from apbt import APBT


def main():
    '''main of the program'''

    debugging = True
    population_size = 80 # population size, 80 good number
    epochs = 1000 # number of epochs



    # run trials
    for trial in range(3):
        apbt = APBT(
            population_size,
            epochs,
            debugging
        )


        # set log path
        apbt.log_path= f'logs/mnist{trial}.csv'
    
        print('Trial: ', trial)
        print('\nRunning the population based training\n')
        best_net, most_acc = apbt.train()
        print('\nPopulation Based Training complete\n')
        # create the artificial neural network

        # printing the neural network
        print('\nPrinting learned weights of best\n')
        best_net.print_network()
        # test the artificial neural network
        print('\nTesting the NN...\n')
        accuracy = 100 * best_net.test(apbt.testing)
        n_params = best_net.num_params()
        print('\nTesting complete\n')
        print(f'\nAccuracy: {accuracy:.2f}%\n')
        print(f'Number of parameters: {n_params}\n')

        # printing the neural network
        print('\nPrinting learned weights of the most accurate\n')
        most_acc.print_network()
        # test the artificial neural network
        print('\nTesting the NN...\n')
        accuracy = 100 * most_acc.test(apbt.testing)
        n_params = most_acc.num_params()
        print('\nTesting complete\n')
        print(f'\nAccuracy: {accuracy:.2f}%\n')
        print(f'Number of parameters: {n_params}\n')
    
if __name__ == '__main__':
    main()

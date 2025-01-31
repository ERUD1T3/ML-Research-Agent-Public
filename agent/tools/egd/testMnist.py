# imports
from egd import EGD


def main():

    '''main of the program'''

    debugging = True
    population_size = 3 # population size, 80 good number
    generations = 20 # number of epochs




    # run trials
    for trial in range(1):
        egd = EGD(population_size,generations,debugging)
        # set log path
        egd.log_path= f'logs/mnist{trial}.csv'

    

        print('Trial: ', trial)
        print('\nRunning the population based training\n')
        best_net, most_acc = egd.train()
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

############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assginment: Term Paper
#   Date: 4/10/2022
#   file: utils.py
#   Description: 
#############################################################

from email import header
import random


def corrupt_data(data, classes, percent):
    '''corrupt the class labels of training examples from 0% to 20% (2% in-
    crement) by changing from the correct class to another class; output the
    accuracy on the uncorrupted test set with and without rule post-pruning.'''

    # get the number of training examples
    num_examples = len(data)
    # get the number of classes to corrupt
    num_examples_to_corrupt = int(percent * num_examples)
    # get the elements to corrupt
    corrupt_elements = random.sample(range(num_examples), num_examples_to_corrupt)

    # corrupt the data
    for e in corrupt_elements:
        # get the class label
        correct_label = data[e][-1]
        
        random_class = random.choice(classes)

        # while the random class is the same as the correct class
        while random_class == correct_label:
            random_class = random.choice(classes)
    
        # change the class label
        data[e][-1] = random_class
        
    return data

def log_csv(path: str, histories: list, headers: list) -> None:
    """Log training histories to a CSV file.
    
    Creates necessary directories if they don't exist and writes training metrics
    to a CSV file with epoch numbers and provided headers.
    
    Args:
        path: Path to the output CSV file
        histories: List of lists containing metric histories to log
        headers: List of column headers for the metrics
        
    Returns:
        None
    """
    import os
    
    # Create directory path if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Add epoch column header
    headers = ['e'] + headers
    
    # Open file and write data
    try:
        with open(path, 'w') as f:
            # Write CSV headers
            f.write(','.join(headers) + '\n')
            
            # Write each epoch's metrics
            for epoch in range(len(histories[0])):
                # Start line with epoch number
                line = f'{epoch},'
                
                # Add each metric value
                for metric_history in range(len(histories)):
                    line += str(histories[metric_history][epoch]) + ','
                    
                # Remove trailing comma and write line
                f.write(line[:-1] + '\n')
                
    except IOError as e:
        print(f"Error writing to {path}: {e}")
        raise
    # epoch is number of every line
    



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
import os
from datetime import datetime


def corrupt_data(data: list, classes: list, percent: float) -> list:
    """Corrupt class labels in training data by randomly changing labels.
    
    Takes a dataset and randomly changes the class labels for a specified percentage
    of examples to different valid class labels. This is useful for testing model
    robustness to noisy/corrupted training data.
    
    Args:
        data: List of training examples, where each example's last element is the class label
        classes: List of valid class labels that can be assigned
        percent: Float between 0 and 1 indicating what fraction of examples to corrupt
        
    Returns:
        List containing the corrupted dataset with modified class labels
    """
    # Calculate number of examples to corrupt based on percentage
    num_examples = len(data)
    num_examples_to_corrupt = int(percent * num_examples)
    
    # Randomly select indices of examples to corrupt
    corrupt_elements = random.sample(range(num_examples), num_examples_to_corrupt)

    # Corrupt the selected examples by changing their class labels
    for e in corrupt_elements:
        # Get the current class label
        correct_label = data[e][-1]
        
        # Select a random different class label
        random_class = random.choice(classes)
        while random_class == correct_label:
            random_class = random.choice(classes)
    
        # Replace the original label with the corrupted one
        data[e][-1] = random_class
        
    return data

def log_csv(path: str, histories: list, headers: list, plot: bool = False) -> None:
    """Log training histories to a CSV file and optionally plot metrics.
    
    Creates necessary directories if they don't exist and writes training metrics
    to a CSV file with generation numbers and provided headers. Can also generate
    plots of each metric over generations.
    
    Args:
        path: Path to the output CSV file
        histories: List of lists containing metric histories to log
        headers: List of column headers for the metrics
        plot: Whether to generate plots of metrics (default: False)
        
    Returns:
        None
    """
    
    # Get timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base filename with timestamp
    base_path = os.path.splitext(path)[0]
    base_name = os.path.basename(base_path)
    timestamped_base = f"{base_name}_{timestamp}"
    
    # Update CSV path with timestamp
    csv_path = os.path.join(os.path.dirname(path), f"{timestamped_base}.csv")
    
    # Create directory path if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Add generation column header
    headers = ['gen'] + headers
    
    # Open file and write data
    try:
        with open(csv_path, 'w') as f:
            # Write CSV headers
            f.write(','.join(headers) + '\n')
            
            # Write each generation's metrics
            for gen in range(len(histories[0])):
                # Start line with generation number
                line = f'{gen},'
                
                # Add each metric value
                for metric_history in range(len(histories)):
                    line += str(histories[metric_history][gen]) + ','
                    
                # Remove trailing comma and write line
                f.write(line[:-1] + '\n')
                
        if plot:
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = os.path.join(os.path.dirname(path), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot each metric
            generations = range(len(histories[0]))
            for i, header in enumerate(headers[1:]):  # Skip gen column
                plt.figure(figsize=(10, 6))
                plt.plot(generations, histories[i])
                plt.title(f'{header} vs Generations')
                plt.xlabel('Generation')
                plt.ylabel(header)
                plt.grid(True)
                
                # Save plot
                plot_path = os.path.join(plots_dir, f"{timestamped_base}_{header}.png")
                plt.savefig(plot_path)
                plt.close()
                
    except IOError as e:
        print(f"Error writing to {csv_path}: {e}")
        raise



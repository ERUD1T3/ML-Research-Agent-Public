############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assginment: Term Paper
#   Date: 4/10/2022
#   file: utils.py
#   Description: 
#############################################################

import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np


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

def log_csv(
        path: str, 
        histories: list, 
        headers: list, 
        generations: int, 
        epochs: int, 
        plot: bool = False) -> None:
    """Log training histories to a CSV file and optionally plot metrics.
    
    Creates necessary directories if they don't exist and writes training metrics
    to a CSV file with generation numbers and provided headers. Can also generate
    plots of each metric over epochs, marking generation boundaries. The metric values
    are assumed constant across all epochs within their generation.
    
    Args:
        path: Path to the output CSV file
        histories: List of lists containing metric histories to log (per generation)
        headers: List of column headers for the metrics
        generations: Number of generations run
        epochs: Number of epochs per generation
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
            # Create plots directory
            plots_dir = os.path.join(os.path.dirname(path), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create epoch points for x-axis (all epochs)
            epoch_points = range(generations * epochs)
            
            # Plot each metric
            for i, header in enumerate(headers[1:]):  # Skip gen column
                plt.figure(figsize=(10, 6))
                
                # Expand generation values across their epochs
                epoch_values = []
                for gen_value in histories[i]:
                    # Repeat each generation's value for all its epochs
                    epoch_values.extend([gen_value] * epochs)
                
                plt.plot(epoch_points, epoch_values)
                
                # Add vertical lines for generation boundaries
                for gen in range(generations):
                    plt.axvline(x=gen*epochs, color='gray', linestyle='--', alpha=0.5)
                
                plt.title(f'{header} vs Epochs (with Generation Boundaries)')
                plt.xlabel('Epochs')
                plt.ylabel(header)
                plt.grid(True)
                
                # Add generation labels
                for gen in range(generations):
                    plt.text(gen*epochs, plt.ylim()[0], f'Gen {gen}', 
                            rotation=90, verticalalignment='bottom')
                
                # Save plot
                plot_path = os.path.join(plots_dir, f"{timestamped_base}_{header}.png")
                plt.savefig(plot_path)
                plt.close()
                
    except IOError as e:
        print(f"Error writing to {csv_path}: {e}")
        raise


def show_results_streamlit(csv_path: str = "C:/Users/the_3/Documents/github/ML-Research-Agent-Public/logs/mnist0.csv"):
    """Display training results in a Streamlit interface.
    
    Creates an interactive dashboard showing training metrics and plots using Streamlit.
    Allows toggling between viewing results per generation or expanded across epochs.
    
    Args:
        csv_path: Path to the CSV file containing training metrics
        
    Returns:
        None
    """
    
    st.title('EGD Training Results Dashboard')
    
    # Read CSV data
    try:
        df = pd.read_csv(csv_path)
        
        # Display summary statistics
        st.header('Summary Statistics')
        st.dataframe(df.describe())
        
        # Display raw data table
        st.header('Raw Data')
        st.dataframe(df)

        # Add toggle for generations vs epochs view
        view_mode = st.radio(
            "Select View Mode",
            ["By Generation", "By Epoch"]
        )

        # Default number of epochs
        epochs = 150  # Set based on actual epochs used
        
        # Create x-axis based on view mode
        if view_mode == "By Generation":
            x_values = df['e']  # Use 'e' column for generation number
            x_label = 'Generation'
            values_dict = {col: df[col].values for col in df.columns if col != 'e'}
        else:
            # Expand each generation value across its epochs
            x_values = np.arange(len(df) * epochs)
            x_label = 'Epoch'
            values_dict = {}
            for col in df.columns:
                if col != 'e':
                    expanded = []
                    for val in df[col]:
                        expanded.extend([val] * epochs)
                    values_dict[col] = expanded

        # Plot metrics
        st.header('Training Metrics')
        
        # Create tabs for different metric categories
        tab1, tab2, tab3 = st.tabs(['Accuracy Metrics', 'Network Metrics', 'Hyperparameters'])
        
        with tab1:
            st.subheader('Accuracy Metrics')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_values, values_dict['top'], label='Top Accuracy')
            ax.plot(x_values, values_dict['eff'], label='Effective Accuracy')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Add line chart using Streamlit
            st.line_chart(df[['top', 'eff']])
            
        with tab2:
            st.subheader('Network Performance')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            ax1.plot(x_values, values_dict['perf'], color='green')
            ax1.set_ylabel('Performance')
            ax1.grid(True)
            
            ax2.plot(x_values, values_dict['size'], color='orange')
            ax2.set_ylabel('Network Size')
            ax2.set_xlabel(x_label)
            ax2.grid(True)
            st.pyplot(fig)
            
            # Add line charts using Streamlit
            st.line_chart(df['perf'])
            st.line_chart(df['size'])
            
        with tab3:
            st.subheader('Hyperparameter Evolution')
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            ax1.plot(x_values, values_dict['lr'], color='blue')
            ax1.set_ylabel('Learning Rate')
            ax1.grid(True)
            
            ax2.plot(x_values, values_dict['m'], color='purple')
            ax2.set_ylabel('Momentum')
            ax2.grid(True)
            
            ax3.plot(x_values, values_dict['d'], color='red')
            ax3.set_ylabel('Decay')
            ax3.set_xlabel(x_label)
            ax3.grid(True)
            st.pyplot(fig)
            
            # Add line charts using Streamlit
            col1, col2, col3 = st.columns(3)
            with col1:
                st.line_chart(df['lr'])
            with col2:
                st.line_chart(df['m'])
            with col3:
                st.line_chart(df['d'])
            
        # Add download button for CSV
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="training_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")

if __name__ == "__main__":
    show_results_streamlit()

import datetime
import csv
import os

def log_training_results(epochs, batch_size, learning_rate, optimizer_name, 
                         best_accuracy, model_description, additional_info=""):
    """
    Log the results of a training run to a CSV file.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size used
        learning_rate: Learning rate used
        optimizer_name: Name of the optimizer (e.g., 'Adam', 'SGD')
        best_accuracy: Best test accuracy achieved (percentage)
        model_description: Brief description of the model architecture
        additional_info: Any additional information to log
    """
    # Create log directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Define the log file path
    log_file = 'logs/training_history.csv'
    
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(log_file)
    
    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the data row
    row = {
        'Timestamp': timestamp,
        'Epochs': epochs,
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Optimizer': optimizer_name,
        'Best Test Accuracy (%)': f"{best_accuracy:.2f}",
        'Model Description': model_description,
        'Additional Info': additional_info
    }
    
    # Write to CSV
    with open(log_file, mode='a', newline='') as file:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    print(f"Results logged to {log_file}")
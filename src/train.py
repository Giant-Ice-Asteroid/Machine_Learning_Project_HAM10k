import os
import torch
import time
import copy
from datetime import datetime
import matplotlib.pyplot as plt


####  Training Loop with Logging ####


# Create a directory for logs and checkpoints
def create_log_dir():
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S')) #Creates a uniquely named directory based on the current date and time (e.g., "logs/20250506_142030")
    os.makedirs(log_dir, exist_ok=True) #Creates this directory if it doesn't exist (exist_ok=True prevents errors if it already exists)
    return log_dir

# Function to log metrics to a file
# Takes performance metrics as input
# Opens the log file in 'append' mode ('a') and Writes a new line with comma-separated values (CSV format)
# The :.4f format specifier rounds numbers to 4 decimal places
def log_metrics(log_file, epoch, train_loss, train_acc, val_loss, val_acc):
    with open(log_file, 'a') as f:
        f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")




# Function to plot training history
def plot_training_history(history, log_dir):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1) # Creates a 1×2 grid of subplots (two side-by-side plots). Third number (1): Which subplot position to activate (counting left-to-right, top-to-bottom)
    plt.plot(history['train_loss'], label='Training Loss') # Draws lines for training and validation metrics, 
    plt.plot(history['val_loss'], label='Validation Loss') # Each point represents one epoch
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_history.png'))
    plt.show()
    
    
################## TRAINING FUNCTION ############

# Function to train the model. Parameters:
# model: The neural network we're training
#optimizer: Update algorithm (Adam)
# scheduler: Learning rate adjustment strategy
# dataloaders: Dictionary with 'train' and 'val' data loaders
# dataset_sizes: Number of samples in each dataset
# num_epochs: How many complete passes through the data to perform (default=25)
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=20, log_dir=None):
    
    if log_dir is None:
        log_dir = create_log_dir()
        
    # Create a log file
    #Opens a new file in 'write' mode ('w') which will overwrite any existing file
    # Writes the header row with column names
    log_file = os.path.join(log_dir, 'training_log.csv')
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")    
    
    since = time.time() #since = time.time(): Records the start time to measure total training duration
    
    # Initialize history dictionaries ->  Empty lists that will store metrics for each epoch
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # best_model_wts: A deep copy of the model's current parameters (weights and biases) wts(=weights)
    # --> deep copy in Python creates a new object and recursively copies all objects found in the original, ensuring that changes to the new object do not affect the original
    # model.state_dict() returns these parameters as a dictionary -> We save these to restore the best version later
    # best_acc: Tracks the highest validation accuracy seen so far
    best_model_wts = copy.deepcopy(model.state_dict()) 
    best_acc = 0.0
    
    # Epoch Loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Training/Validation Phase Loop
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode: Activates dropout layers, Enables batch normalization updates and Tracks gradients for parameter updates
            else:
                model.eval()   # Set model to evaluate mode: Disables dropout (uses all neurons), Uses fixed batch normalization statistics and Doesn't track gradients (saves memory)
            
            running_loss = 0.0 #Accumulates the loss value across all batches in this phase
            running_corrects = 0 # Counts total number of correct predictions in this phase
            batch_count = 0
            total_batches = len(dataloaders[phase])
            
            # Progress tracking
            last_progress_update = time.time()
            update_interval = 5  # Print progress every 5 seconds
            
            # Batch Loop
            # Processes one batch of data at a time
            # Moving to device: Transfers the input images and their labels to the GPU (if available)
            for inputs, labels in dataloaders[phase]:
                
                # Print progress for every batch to see where it might be stuck
                batch_count += 1
                
                # Print progress update every 5 seconds instead of for every batch
                current_time = time.time()
                if current_time - last_progress_update >= update_interval:
                    progress = batch_count / total_batches * 100
                    print(f"  {phase} progress: {batch_count}/{total_batches} batches ({progress:.1f}%)")
                    last_progress_update = current_time
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # optimizer.zero_grad(): Clears the gradients from the previous batch
                # Without this, gradients would accumulate and cause incorrect updates
                optimizer.zero_grad()
                
                forward_start = time.time()
                # torch.set_grad_enabled(phase == 'train'):Context manager that Tracks gradients during training phase + Disables gradient tracking during validation (saves memory and computation)
                # outputs = model(inputs): Forward pass - the model processes the batch of images ->  Returns raw scores for each class (logits)
                # _, preds = torch.max(outputs, 1): Gets the predicted class for each image. 
                # --> underscore _ is Python convention for a "throwaway variable" 
                # --> "1" means we find the maximum along dimension 1 (the class dimension). The indices tell us which class received the highest score (0-6)
                # --> torch.max returns both maximum values and their indices
                # --> We only keep the indices (second return value) as our predictions
                # loss = criterion(outputs, labels): Calculates how wrong our predictions are -> CrossEntropyLoss compares model outputs with true labels
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only in training phase
                    # loss.backward(): Calculates gradients for all parameters -> This is backpropagation - determining how to adjust each parameter
                    # optimizer.step(): Updates model parameters based on gradients -> The Adam optimizer uses calculated gradients to adjust weights and biases
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                forward_time = time.time() - forward_start
                print(f"    Forward/backward pass took {forward_time:.2f}s")
                # Batch Statistics Accumulation
                #loss.item() gets the scalar value of the loss
                # Multiplied by batch size (inputs.size(0)) to get total loss -> We add this to our running total
                running_loss += loss.item() * inputs.size(0)
                #preds == labels.data creates a tensor of boolean values (True where predictions are correct)
                # torch.sum() counts the number of True values (correct predictions) -> We add this to our running total
                running_corrects += torch.sum(preds == labels.data)
            
            #Learning Rate Update
            # scheduler.step(): Updates the learning rate according to the schedule
            if phase == 'train':
                scheduler.step()
            
            # Phase Statistics Calculation
            # epoch_loss = running_loss / dataset_sizes[phase]: Calculates the average loss per sample
            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]: Calculates the proportion of correct predictions
            # .double() converts to double precision for division
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # history updates: Stores the metrics in our history dictionary
            # .item() extracts the scalar value from a single-element tensor
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}') # Shows progress for this phase
            
            #Best Model Checkpoint
            # Deep copy the model if it's the best
            # if phase == 'val' and epoch_acc > best_acc: Checks if this is the best validation accuracy so far            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc # Updates the best accuracy record
                best_model_wts = copy.deepcopy(model.state_dict()) # Saves a deep copy of the current model state
                
                # Save the best model -> ensures we can recover the best-performing model even if it occurs early in training
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                }, os.path.join(log_dir, 'best_model.pt'))
        
        # Logging and Regular Checkpoints
        # log_metrics: Writes the epoch's metrics to our CSV file
        # -1 indices access the most recently added values in the history lists
        log_metrics(
            log_file,
            epoch,
            history['train_loss'][-1],
            history['train_acc'][-1],
            history['val_loss'][-1],
            history['val_acc'][-1]
        )
        
        # Periodic checkpointing: Saves the model state every 5 epochs
        # Useful for resuming training if it's interrupted and Allows going back to intermediate models if needed
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        print()
    
    # Training Wrap-up
    # time_elapsed: Calculates total training time -> // is integer division for minutes; % is remainder for seconds
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Loading best weights: Restores the model to its best-performing state
    # This is why we kept a copy of the best weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plot_training_history(history, log_dir)
    
    return model, history, log_dir # Provides access to both the trained model and the training history


#### Prepare for training


"""
# dataloaders dictionary: Organizes data loaders by phase for easy access
dataloaders = {
    'train': train_loader,
    'val': val_loader
}

#dataset_sizes: Stores the number of samples in each dataset
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

# Train the model
num_epochs = 20 # num_epochs=20: Sets how many complete passes through the training data
model, history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=num_epochs) # : Calls our training function and stores results
"""
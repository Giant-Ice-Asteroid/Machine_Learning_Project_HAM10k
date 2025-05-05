import json
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from network import NeuralNetwork
from log_results import log_training_results

# Define base directory -Sets the base directory to "data"
base_skin_dir = os.path.join('data')

# Use glob to find all jpg images in any subfolder of lesion_images
# The * wildcard will match both HAM10000_images_part_1 and HAM10000_images_part_2
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, 'lesion_images', '*', '*.jpg'))}

# Print some stats to verify it's working
print(f"Total images found: {len(imageid_path_dict)}")

# Print a few examples
print("\nSample entries:")
sample_count = 0
for name, path in imageid_path_dict.items():
    print(f"{name}: {path}")
    sample_count += 1
    if sample_count >= 5:
        break

#dict with dx spelled out
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
# Load the metadata
try:
    skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
    
    # Map image paths and cell types to the dataframe
    skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
    skin_df['lesion_type'] = skin_df['dx'].map(lesion_type_dict.get)
    skin_df['lesion_type_idx'] = pd.Categorical(skin_df['lesion_type']).codes
    
    # Display the head of the dataframe
    print("\nDataframe head:")
    print(skin_df.head())
    
except FileNotFoundError:
    print("\nMetadata file not found. Skipping dataframe creation.")
    
# Load model configuration
with open('config.json', 'r') as f:
    config = json.load(f)

EPOCHS = config['epochs']
LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['batch_size']

   
###  Training loop #####

def train_loop(dataloader, model, loss_fn, optimizer, device):
    """Train the model for one epoch"""
    # Set model to training mode
    model.train()

    # Variables to track progress
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    correct = 0

    # Process each batch of images
    # X contains the images (a batch of them)
    # y contains the correct labels (which clothing type each image is)
    for batch, (X, y) in enumerate(dataloader):
        # Move data to the right device (CPU or GPU)
        X, y = X.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(X) #model processes the batch of images and makes predictions
        loss = loss_fn(pred, y) #Measures how wrong the predictions are compared to the true labels
        train_loss += loss.item()
        
        # Count correct predictions
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Backpropagation (learning part)
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute how each weight in the network affects the error (gradients)
        optimizer.step()       # Update weights  to reduce the error
        
        # Print progress
        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

    # Calculate final statistics
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    
    
### Testing loop ###


def test_loop(dataloader, model, loss_fn, device):
    """Evaluate the model on the test dataset."""
    # Set model to evaluation mode
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    # No gradient calculation needed for testing (saves nmemory)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) # The model predicts the clothing type for each test image
            test_loss += loss_fn(pred, y).item() #We count how many predictions were correct
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #pred.argmax(1): Finds the clothing class with the highest score for each image

    # Calculate final statistics
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct  # Return accuracy for tracking
   
class HAM10KDS(skin_df): # custom dataset
    
    def __init__(self, df, transforms=None, selective=True, OR=4, normalised=True):
        self.data = df # the dataframe containing the image
        self.y = self.data.label # label y
        self.x = self.data.drop(labels='label', axis=1) # drop the label, get images
        
        # Reshape the entire image array into N x height x width x RGB channel. N = number of samples
        self.x = np.asarray(self.x).reshape(-1, 28, 28,3)
        
        # Calculate the RGB stats for normalisation, and convert to tensor.
        self.mean = torch.tensor([np.mean(self.x[:, :, :, channel]) for channel in range(3)]).reshape(1, 3, 1, 1) 
        self.std = torch.tensor([np.std(self.x[:, :, :, channel]) for channel in range(3)]).reshape(1, 3, 1, 1)  
        
        # Convert images to torch tensor for transforms and training on GPU. Images will be moved to GPU outside dataloader.
        self.x = torch.tensor(self.x, dtype=torch.float32).permute(0, 3, 1, 2)
        
        # Normalise the images with the stats calculated
        if normalised==True:
            self.x -= self.mean
            self.x /= self.std
        print('images are normalised')
        
        self.resize = transforms.Resize((28*4, 28*4)) # Increase images by 4 in both dimension.

        self.OR = OR # the label of the over represented class
        self.tf = transforms # the input list of transforms
        self.selective = selective # flag to apply transform to under rep classes only
        
    def __len__(self):
        return len(self.data) # number of samples in dataset
    
    def __getitem__(self,idx):
        label = torch.tensor(self.y.iloc[idx]) # get label of image and convert to torch tensor
        
        img = self.x[idx] # get the image from the big tensor
        img = self.resize(img) # upsize the image
        
        # Applying transforms
        if self.tf!=None:
            if self.selective==True: # Can choose to NOT apply augmentation on over rep classes
                if label.item()!=self.OR: 
                    img = self.tf(img)
            else: # Or just apply aug to ALL classes and samples
                img = self.tf(img)
                                
        return img, label
    
## preparing dataframe for model.. ###

# First, create a train+validation set and a test set (e.g., 80/20 split)
train_val_df, test_df = train_test_split(
    skin_df, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    stratify=skin_df["lesion_type"]  # Maintain class distribution
)

# Then split train+validation into separate train and validation sets (e.g., 75/25 split)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,          # 25% of 80% = 20% of original data
    random_state=42,
    stratify=train_val_df["lesion_type"]
)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Verify class distribution is maintained
print("\nClass distribution in original data:")
print(skin_df["lesion_type"].value_counts(normalize=True))

print("\nClass distribution in training set:")
print(train_df["lesion_type"].value_counts(normalize=True))





#### MAIN ###

def main(): 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Using device: {device}")
    

    # Additionally applies normalization using the statistics we just calculated
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load the datasets with the correct normalization
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',              # Same location as training data
    train=False,                # This is testing data
    download=True,              # Download if needed
    transform=transform         # Same transformation
    )
    
    # Create data loaders
    # Creates an iterator that delivers batches of data during training (to not use too much memory at once)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,      # Our training dataset
        batch_size=BATCH_SIZE,      # Process 64 images at once (from config)
        shuffle=True                # Shuffle data to avoid learning the order
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,       # Our testing dataset
        batch_size=BATCH_SIZE,      # Same batch size for testing
        shuffle=False               # No need to shuffle test data
    )
    
    # Print dataset information
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Print the class labels (what each number represents)
    #  the 10 categories of clothing items in our dataset (known beforehand)
    #Each image in the dataset has a label from 0 to 9, corresponding to these classes
    #Our neural network will predict which of these 10 classes each image belongs to
    #classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
      #         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #print(f"Classes: {classes}")
        ### Model instantiation ###
    
    # Create model and move it to the right device (CPU or GPU)
    model = NeuralNetwork().to(device)
    print(f"Model structure: {model}")
    
    # Define loss function and optimizer
    # A function that measures how wrong the model's predictions are
    # CrossEntropyLoss: Specifically designed for classification problems like ours
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer Updates the model's weights based on the loss
    # SGD (Stochastic Gradient Descent): A method for adjusting weights to reduce errors
    # Parameters: These are the weights of the neural network that are being adjusted
    # Learning rate: Controls how big each adjustment is
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # opting for Adam (Adaptive Moment Estimation) rather than SGD to optimize to optimizer:) ->
    # Adam adapts the learning rate for each parameter based on historical gradients (SGD uses the same learning rate)
    # Momentum and Bias Correction -> SGD: Basic version has no momentum while Adam has built-in momentum and bias correction mechanisms
    # Parameter Updates -> SGD adjusts weights proportionally to the gradient, Adam maintains two moving averages (first+second moments) to optimize the update step size
        
    # Training and testing
    best_accuracy = 0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, device)
        accuracy = test_loop(test_loader, model, loss_fn, device)
        
        # Save the model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "HAM10000_model.pt")
            print(f"Model saved with accuracy: {(100*best_accuracy):>0.1f}%")
            
 
    print("Training complete!")
    print(f"Best accuracy: {(100*best_accuracy):>0.1f}%") 
    
    # Log the final results
    optimizer_name = type(optimizer).__name__  # Gets name like 'Adam' or 'SGD'
    model_description = "Simple Fully Connected Network (784-512-10)"

    
    log_training_results(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        optimizer_name=optimizer_name,
        best_accuracy=best_accuracy * 100,  # Convert to percentage
        model_description=model_description,
    )

    # Load the best model for visualization
    best_model = NeuralNetwork().to(device)
    best_model.load_state_dict(torch.load("HAM10000_model.pt"))

    print(model)


if __name__ == "__main__":
    main()
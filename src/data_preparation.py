import os
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
# Fix the import by using relative import
try:
    # When running as a module
    from .dataset_class import SkinLesionDataset
except ImportError:
    # When running directly
    from dataset_class import SkinLesionDataset



# Define base directory -Sets the base directory to "data"
base_skin_dir = os.path.join("data")


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


############ TRANSFORMATIONS #################
#In PyTorch, transformations are operations that modify data (= images) before feeding them to the model. 
# They're not a class or method specifically, but rather a collection of functions from the torchvision.transforms module
# 

# define data transformations
#transforms.Compose is a wrapper that chains multiple transforms together. When you apply this composed transform to an image, each transformation is applied in sequence.
# note that aining transforms include augmentations, while validation and test transforms don't. This is because:
# Training: We want to expose the model to variations (flips, rotations, etc.) to improve generalization
# Validation/Testing: We want to evaluate the model on clean, unmodified images that represent real-world data
# fundamental principle in ML
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Takes an image of any size and resizes it to 224×224 pixels: ensures all images have same dimensions for batch processing
        transforms.RandomHorizontalFlip(),  # Flips the image horizontally with a 50% probability -> model learns that horizontal orientation doesn't change the diagnosis
        transforms.RandomVerticalFlip(),    # as above, just flips the image vertically with a 50% probability
        transforms.RandomRotation(20),      # Rotates the image by a random angle between -20 and +20 degrees->model learns that rotation doesn't affect classification
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  #Randomly changes brightness, contrast, saturation, hue->simulates diff lighting conditions and camera settings
        transforms.ToTensor(), #Converts a PIL Image or numpy array to a PyTorch tensor->  Converts pixel values from 0-255 to 0-1 and changes dimensions from (height, width, channels) to (cs, h, w)
        #Normalizes tensor images with mean and standard deviation. Under the hood: Applies the formula: (pixel - mean) / std for each channel (vals are from ImageNET)
        # First parameter [0.485, 0.456, 0.406]: These are the mean values for each channel (R, G, B)
        # Second parameter [0.229, 0.224, 0.225]: These are the standard deviation values for each channel
        # These specific values are the per-channel means and standard deviations calculated across all images in the ImageNet dataset ( 1.2+ million images)
        #  we use these specific values: we're using a ResNet-18 model pre-trained on ImageNet, work well across many computer vision tasks, standard practice
        # When we normalize with these values, we're applying this formula to each pixel in each channel: normalized_pixel = (original_pixel - channel_mean)/channel_standard_deviation
        # Normalized inputs tend to help models converge faster during training
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Use glob to find all jpg images in any subfolder of lesion_images
# The * wildcard will match both HAM10000_images_part_1 and HAM10000_images_part_2
def create_image_path_dict(base_dir=base_skin_dir):
    """Create a dictionary mapping image IDs to file paths"""
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(base_dir, 'lesion_images', '*', '*.jpg'))}
    return imageid_path_dict


"""
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

"""

# Load the metadata
def load_metadata(base_dir=base_skin_dir):
    """Load metadata and map paths and classes"""
    imageid_path_dict = create_image_path_dict(base_dir)
    
    try:
        skin_df = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))
        
        # Map image paths and cell types to the dataframe
        skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
        skin_df['lesion_type'] = skin_df['dx'].map(lesion_type_dict.get)
        skin_df['lesion_type_idx'] = pd.Categorical(skin_df['lesion_type']).codes
    
        # Handle missing paths
        print(f"Images with missing paths: {skin_df['path'].isna().sum()}")
        skin_df = skin_df.dropna(subset=['path'])
        print(f"Dataset size after cleaning: {len(skin_df)}")
        
        return skin_df
    except FileNotFoundError:
        print("\nMetadata file not found.")
        return None

"""
# Examine class distribution
class_counts = skin_df['lesion_type'].value_counts()
print("Class distribution:")
print(class_counts)
print(f"Percentage of 'Melanocytic nevi': {class_counts['Melanocytic nevi'] / len(skin_df) * 100:.2f}%")

# Visualize the class distribution
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar')
plt.title('Class Distribution in HAM10000 Dataset')
plt.xlabel('Lesion Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Display some sample images to understand what we're working with
def show_sample_images(df, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        # Get a random sample from each class
        sample = df.sample(1).iloc[0]
        img_path = sample['path']
        
        if img_path is not None:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"{sample['lesion_type']}\n{img.size}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
"""
# Show samples from each class
"""for lesion_type in skin_df['lesion_type'].unique():
    print(f"Samples of {lesion_type}:")
    class_df = skin_df[skin_df['lesion_type'] == lesion_type]
    show_sample_images(class_df)"""
    
    


    
###### DATA SPLITTING  ############

"""
# Check for and handle NaN values in the path column
print(f"Images with missing paths: {skin_df['path'].isna().sum()}")
skin_df = skin_df.dropna(subset=['path'])
print(f"Dataset size after removing rows with missing paths: {len(skin_df)}")

# Split the data into train, validation, and test sets
# First, split off the test set (20% of data)
# random_state sets the seed for the random number generator,  42 is arbitrary - any integer would work
# stratify=skin_df['lesion_type_idx'] ensures that each split maintains the same class distribution as the original dataset
# without stratification, you might end up with some classes missing entirely from a split
train_val_df, test_df = train_test_split(skin_df, test_size=0.20, random_state=42, stratify=skin_df['lesion_type_idx'])

# Then split the train_val set into train and validation (80% train, 20% validation of the remaining data)
train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=42, stratify=train_val_df['lesion_type_idx'])

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")
"""
def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, 
        stratify=df['lesion_type_idx']
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state,
        stratify=train_val_df['lesion_type_idx']
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, val_df, test_df

#####  HANDLING CLASS IMBALNCES ############

# Handle class imbalance using oversampling for the training set
# This duplicates samples from minority classes to balance the dataset
def oversample_minority_classes(df, target_column):
    # Get the class counts
    class_counts = df[target_column].value_counts()
    max_class_count = class_counts.max()
    
    # Create a list to hold the balanced dataframe
    balanced_dfs = []
    
    # Loops through each class and its count (e.g., 'nevi': 7000, 'melanoma': 1000, etc.
    for class_name, count in class_counts.items():
        # Creates a subset of the dataframe containing only rows of the current class
        # i.e if class_name is 'melanoma', class_df contains only melanoma samples
        class_df = df[df[target_column] == class_name]
                
        #calculates how many samples we need to add to match the majority class
        # i.e if max_class_count is 7000 (nevi) and count is 1000 (melanoma), we need to add 6000 samples
        samples_to_add = max_class_count - count
        
        # If we need to add samples -> only add samples if this is a minority class (not the majority class)
        if samples_to_add > 0:
            
            #Randomly samples from the current class with replacement
            # replace=True means the same sample can be selected multiple times
            # crucial since we're adding more samples than we have in the original class
            oversampled = class_df.sample(samples_to_add, replace=True, random_state=42)
            # Combine the original samples with the oversampled ones
            # e.g for melanoma, we now have the original 1000 samples plus 6000 duplicated samples
            balanced_class_df = pd.concat([class_df, oversampled])
        else:
            # For the majority class, just keep the original samples
            balanced_class_df = class_df
        
        #Adds the balanced class dataframe to a list
        balanced_dfs.append(balanced_class_df)
    
    # Combines all the balanced class dataframes into one
    balanced_df = pd.concat(balanced_dfs)
    

    # Shuffles the rows randomly (frac=1 means use 100% of the data), and resets the index so it starts from 0
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

"""
# Apply oversampling to the training data only
balanced_train_df = oversample_minority_classes(train_df, 'lesion_type')
print(f"Balanced training set size: {len(balanced_train_df)}")

# Verify the class distribution in the balanced training set
balanced_class_counts = balanced_train_df['lesion_type'].value_counts()
print("Balanced class distribution:")
print(balanced_class_counts)
"""


"""
# Create the datasets
# The Dataset class is where transformations are actually applied to the data
# When we create instances of our SkinLesionDataset, we pass in the appropriate transforms:
train_dataset = SkinLesionDataset(balanced_train_df, transform=data_transforms['train'])
val_dataset = SkinLesionDataset(val_df, transform=data_transforms['val'])
test_dataset = SkinLesionDataset(test_df, transform=data_transforms['test'])

# Create data loaders
# DataLoader is a function from the torch library 
# The DataLoader class:
# -> Wraps a Dataset and efficiently loads batches of data
# -> Handles batching, shuffling, and parallel data loading
# -> Manages multiple "workers" (=parallel processes) to load data faster
# When the DataLoader function requests a sample from the Dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
"""

def prepare_data(batch_size=32, num_workers=0):
    """Complete pipeline to prepare datasets and dataloaders"""
    # Load data
    skin_df = load_metadata()
    if skin_df is None:
        return None, None, None, None, None, None, None
    
    # Split data
    train_df, val_df, test_df = split_data(skin_df)
    
    # Balance training data
    balanced_train_df = oversample_minority_classes(train_df, 'lesion_type')
    
    # Create datasets
    train_dataset = SkinLesionDataset(balanced_train_df, transform=data_transforms['train'])
    val_dataset = SkinLesionDataset(val_df, transform=data_transforms['val'])
    test_dataset = SkinLesionDataset(test_df, transform=data_transforms['test'])
    
    # Print some dataset shapes for debugging
    print(f"Test dataset length: {len(test_dataset)}")
    test_img, test_label = test_dataset[0]
    print(f"Sample image shape: {test_img.shape}, label: {test_label}")
    
    # Create data loaders with smaller batch size for visualizations
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Use a smaller batch size for test loader to avoid issues during visualization
    test_batch_size = min(batch_size, 4)  # Ensure small batches for test loader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    
    return skin_df, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
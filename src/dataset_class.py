
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Custom dataset Class that inherits form Pytorch's Dataset class
        
        Arguments:
        
            dataframe: Pandas dataframe containing image paths and labels
            transform: Optional transform to be applied to the images
        """
        self.dataframe = dataframe
        self.transform = transform
    
    #returns the dataset size   
    def __len__(self):
        return len(self.dataframe)
    
    #used to load images and labels on-demand
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the path and label for this index
        img_path = self.dataframe.iloc[idx]['path']
        
        # Handle cases where the path might be missing
        if img_path is None or pd.isna(img_path):
            # Create a blank image as a fallback
            image = Image.new('RGB', (224, 224), color='black')
        else:
            # Open the image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Get the label
        label = self.dataframe.iloc[idx]['lesion_type_idx']
        label = torch.tensor(int(label), dtype=torch.long)  # Convert to integer then to torch.long tensor
        
        return image, label
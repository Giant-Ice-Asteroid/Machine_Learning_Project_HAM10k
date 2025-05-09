import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt  # Fixed import - was missing pyplot
from PIL import Image

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

# Use glob to find all jpg images in any subfolder of lesion_images
# The * wildcard will match both HAM10000_images_part_1 and HAM10000_images_part_2
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
for x in glob(os.path.join(base_skin_dir, 'lesion_images', '*', '*.jpg'))}

skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

# Map image paths and cell types to the dataframe
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['lesion_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['lesion_type_idx'] = pd.Categorical(skin_df['lesion_type']).codes

# Handle missing paths
print(f"Images with missing paths: {skin_df['path'].isna().sum()}")
skin_df = skin_df.dropna(subset=['path'])
print(f"Dataset size after cleaning: {len(skin_df)}")

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
plt.savefig('class_distribution.png')  # Save figure before showing
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
    plt.savefig(f"samples_{df['lesion_type'].iloc[0].replace(' ', '_')}.png")  # Save each class's samples
    plt.show()

# Show samples from each class
for lesion_type in skin_df['lesion_type'].unique():
    print(f"Samples of {lesion_type}:")
    class_df = skin_df[skin_df['lesion_type'] == lesion_type]
    show_sample_images(class_df)
    
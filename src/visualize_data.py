import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from glob import glob

"""
Data visualization, exploration and cleaning

"""

# exploring the raw metadata file:

meta_to_df = pd.read_csv("data\\HAM10000_metadata.csv")

df = pd.DataFrame(meta_to_df)

df.info()

print(df.describe(include='all'))

#first 5 rows
print(df.head())

#Checking for NA
print("\nChecking for NULL values:")
print(df.isna().isna().sum())



# Define base directory -Sets the base directory to "data"
base_skin_dir = os.path.join('data')

# Use glob to find all jpg images in any subfolder of lesion_images
# The * wildcard will match both HAM10000_images_part_1 and HAM10000_images_part_2
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, 'lesion_images', '*', '*.jpg'))}

# Print some stats to verify it's working
print(f"\nTotal images found: {len(imageid_path_dict)}")

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


# Load the combined metadata
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

df_fig = plt.figure(figsize=(15,10))

#shows number of each dx in dataset
skin_df["lesion_type"].value_counts().plot(kind="bar", ylabel="Count")    
plt.show()    

## note: A sizable amount of the dataset is nevi -> find a way to adjust for this      

#shows physical location of each lesion
skin_df["localization"].value_counts().plot(kind="bar", ylabel="count")
plt.show()

sns.set_theme()

sns.countplot(x="sex", data=skin_df)
plt.show()

sns.countplot(x='lesion_type',hue='sex', data=skin_df, order = skin_df['lesion_type'].value_counts().index)
plt.show()

# to do: age, age/gender, lesion/location, dx_type/lesion, ..


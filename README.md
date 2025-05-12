# Skin Lesion Classification using Machine Learning


A project for classifying skin lesions from dermatoscopic images by building a machine learning model.


## Project Overview
This project implements a convolutional neural network for the classification of skin lesions across 7 different categories using the HAM10000 dataset. The model uses transfer learning with a pre-trained ResNet-18 backbone, achieving 66.1% accuracy on the test set.
Dataset

The model is trained on the HAM10000 ("Human Against Machine with 10000 training images") dataset, consisting of dermatoscopic images of pigmented skin lesions. The dataset includes 10015 images across 7 categories:

Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc)
Actinic keratoses (akiec)
Vascular lesions (vasc)
Dermatofibroma (df)

***

## Set up & Installation

### Prerequisites:

Python 3.7+
PyTorch 1.7+
CUDA-capable GPU (recommended for training, will otherwise take significant amount of time!)

### Setup

Clone the repository:
git clone https://github.com/Giant-Ice-Asteroid/Machine_Learning_Project_HAM10k


Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Download the HAM10000 dataset and organize it as follows:
data/
├── HAM10000_metadata.csv
└── lesion_images/
    ├── HAM10000_images_part_1/
    └── HAM10000_images_part_2/


### Project Structure
├── data/                      # Data directory (not included in repo!!)
├── logs/                      # Training logs and model checkpoints
├── src/                       # Source code
│   ├── data_preparation.py    # Data loading and preprocessing
│   ├── dataset_class.py       # PyTorch dataset implementation
│   ├── evaluate.py            # Model evaluation and visualization
│   ├── main.py                # Main training and evaluation script
│   ├── model.py               # Model architecture definition
│   ├── train.py               # Training loop implementation
│   └── utils.py               # Utility functions
├── runme.py                   # Runner script
├── requirements.txt           # Python dependencies
└── README.md                  # This file

***

## Usage

### Training a Model
To train the model from scratch:
python runme.py --batch_size 16 --num_epochs 25 --lr 0.001

### Evaluating a Model
To evaluate a pre-trained model:
python runme.py --eval_only --pretrained=logs/*YOUR_MODEL_FOLDER*/best_model.pt

### Visualizing Results
The evaluation process automatically generates:

Confusion matrix
Sample predictions (correct and incorrect)
Feature map visualizations
GradCAM visualizations

All visualizations are saved to the logs directory.

***

## Data download:
HAM10000 Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

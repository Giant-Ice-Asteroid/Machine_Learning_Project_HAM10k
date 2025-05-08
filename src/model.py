import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim import lr_scheduler


# Check if CUDA is available and set device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# function to create a model using transfer learning:
# uses a pre-trained ResNet model and modify it for our specific task
# ResNet-18 is a Convolutional Neural Network with a special architecture that includes "residual connections"
# using ResNet18 because it's relatively lightweight (18 layers) but still powerful and known to perfrom well with image classification
# The ResNet18 model consists of 18 layers, including convolutional layers and residual blocks
#  A residual block allows the input to bypass one or more layers via a shortcut connection, which helps in mitigating the vanishing gradient problem.
    
def create_model(num_classes):
    # first load a pre-trained ResNet-18 model
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
        
    # Freeze all the parameters to prevent them from updating 
    # Preserve learned features -> pre-trained model has already learned useful features
    # Prevent overfitting -> with limited data, retraining all parameters might lead to overfitting
    # Efficiency: Training fewer parameters requires less computation
    # Basically: use the pre-trained features but adapt the final classification
    # Freezes: All convolutional layers: The feature extraction pipeline and All batch normalization layers: Keep the normalization statistics
    # --> freezing everything except the final fully connected layer: We're freezing all layers except the classifier we add
    for param in resnet_model.parameters():
        param.requires_grad = False
    
    # This code replaces the original fully connected layer (which was designed for ImageNet classes) with our custom classifier:
    # The last layer has 512 input features and outputs num_classes
    resnet_model.fc = nn.Sequential(
        nn.Linear(512, 256), #First fully connected layer -> 512 input features (ResNet's feature extractor), 256 output features (our custom middle layer size, arbitrary, might adjust)
        nn.ReLU(), # Activation function (non-linearity), allows the network to learn complex patterns
        nn.Dropout(0.5), # Randomly turns off 50% of neurons during trainingto prevennt overfitting <- might adjust to control overfitting tendency
        nn.Linear(256, num_classes) # Final output layer, has 256 input features, num_classes outputs (7 for this dataset) ->Each output = probability of a specific lesion type
    )
    
    # Move the model to the appropriate device
    resnet_model = resnet_model.to(device)
    
    return resnet_model


# Cross-Entropy Loss ("Criterion" is just another term for "loss function" in PyTorch) Measures how well our model is predicting the correct classes
# -> Compares predicted probability distribution with the actual class
# Penalizes confident incorrect predictions more heavily
# Ideal for multi-class classification problems
def create_criterion():
    """Create a Cross-Entropy loss function"""
    return nn.CrossEntropyLoss()


# Adam Optimizer: Updates model weights to minimize the loss
# -> adaptive learning rate for each parameter
# -> Incorporates momentum for faster convergence
# only optimizes model.fc.parameters() because the rest are frozen
# lr=0.001 is the learning rate (step size for updates)

def create_optimizer(model, learning_rate=0.001):
    """Create an Adam optimizer for the model's FC layer"""
    return optim.Adam(model.fc.parameters(), lr=learning_rate)

#Learning Rate Scheduler: Adjusts learning rate during training
# -> Starts with lr=0.001
# -> Every 7 epochs, multiplies the learning rate by 0.1 ->Example: lr=0.001 → 0.0001 → 0.00001
# Large steps at the beginning for faster progress, Small steps later for fine-tuning
def create_scheduler(optimizer, step_size=7, gamma=0.1):
    """Create a learning rate scheduler"""
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
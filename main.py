import os
import torch
import argparse # used for parsing command-line arguments
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
# from my modules
from dataset_class import SkinLesionDataset
from utils import set_seed, create_log_dir, save_config, plot_class_distribution
from dataset_class import SkinLesionDataset
from data_preparation import (
    load_metadata,
    split_data, 
    oversample_minority_classes,
    data_transforms,
    train_loader, 
    val_loader, 
    test_loader,
    skin_df
)
from model import criterion, optimizer, scheduler
from train import history, train_model, dataloaders, dataset_sizes
from evaluate import evaluate_model, accuracy, cm, show_prediction_examples, show_gradcam_examples, visualize_model_features

def main():
    
    # Parse command line arguments -> values provided to the program when started from the command line
    # For each argument the parser specifies the type of value expected, a default value and txt depsrciption
    # Instead of changing the code each time when wanting try different settings, can run for ex.: python main.py --batch_size 64 --lr 0.0005
    # also, Scripts with command line arguments can be easily integrated into larger workflows, automated pipelines, or run from batch files.
    parser = argparse.ArgumentParser(description='Skin Lesion Classification') #creates a parser that will recognize and interpret command line arguments    
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    
    # set random seed
    set_seed(args.seed)
    
    #create the log dir
    log_dir = create_log_dir()
    
    #save configuraiton
    config = vars(args)
    save_config(config, log_dir)

    # device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # STEP 1 #
    # load and preprocess data (using functions from data_preparation.py script)
    print("Using pre-processed data from data_preparation.py")
    
    #get class names for vislualizations
    class_names = skin_df["lesion_type"].unique().tolist()
    
    # STEP 2 #    
    #create model
    #Loads a pre-trained ResNet-18 model-> Freezes all parameters to prevent them from being updated-> Replaces  final fully connected layer with custom classifier
    print("Creating model..")
    num_classes = len(skin_df['lesion_type'].unique())
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    # Move model to device
    model = model.to(device)
    
    # STEP 3 #
    # Training the model
    
    print(f"Training for {args.num_epochs} epochs...")
    model, history = train_model(
        model, criterion, optimizer, scheduler, 
        dataloaders, dataset_sizes, num_epochs=args.num_epochs,
        log_dir=log_dir
    )
    
    # STEP 4 #
    # evaluate the model
    print("Evaluating model...")
    accuracy, report_df, cm = evaluate_model(model, test_loader, class_names)
    report_df.to_csv(os.path.join(log_dir, 'classification_report.csv'))
    
    # STEP 5 # 
    # create visualizations
    print("Visualizing predictions...")
    show_prediction_examples(model, test_loader, class_names)
    
    # Step 11: Visualize features
    print("Visualizing features...")
    visualize_model_features(model, test_loader, class_names)
    
    # Step 12: Grad-CAM visualization
    print("Generating Grad-CAM visualizations...")
    show_gradcam_examples(model, test_loader, class_names)
    
    print(f"Done! All results saved to {log_dir}")
    
if __name__ == "__main__":
    main()
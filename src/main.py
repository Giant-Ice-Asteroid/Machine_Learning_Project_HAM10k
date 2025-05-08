import torch
import argparse # used for parsing command-line arguments
import torch.multiprocessing as mp
# from my modules
from data_preparation import prepare_data
from model import create_model, create_criterion, create_optimizer, create_scheduler, device
from train import train_model
from evaluate import evaluate_model, show_prediction_examples, visualize_model_features, show_gradcam_examples
from utils import set_seed
import time



def main():
    
    # Parse command line arguments -> values provided to the program when started from the command line
    # For each argument the parser specifies the type of value expected, a default value and txt depsrciption
    # Instead of changing the code each time when wanting try different settings, can run for ex.: python main.py --batch_size 64 --lr 0.0005
    # also, Scripts with command line arguments can be easily integrated into larger workflows, automated pipelines, or run from batch files.
    parser = argparse.ArgumentParser(description='Skin Lesion Classification') #creates a parser that will recognize and interpret command line arguments    
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    args = parser.parse_args()
    
    
    # set random seed
    set_seed(args.seed)
    

    # STEP 1 #
    # load and preprocess data (using functions from data_preparation.py script)
    print("Preparing data...")
    start_time = time.time()
    skin_df, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = prepare_data(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Data preparation took {time.time() - start_time:.2f} seconds")
    
    if skin_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare dataloaders dictionary
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    # Step 2: Create Model
    print("Creating model...")
    start_time = time.time()
    num_classes = len(skin_df['lesion_type'].unique())
    model = create_model(num_classes)
    criterion = create_criterion()
    optimizer = create_optimizer(model, learning_rate=args.lr)
    scheduler = create_scheduler(optimizer)
    print(f"Model creation took {time.time() - start_time:.2f} seconds")
    
    
    # Step 3: Train Model
    print(f"Training for {args.num_epochs} epochs...")
    start_time = time.time()
    model, history, log_dir = train_model(
        model, criterion, optimizer, scheduler, 
        dataloaders, dataset_sizes, device,
        num_epochs=args.num_epochs
    )
    print(f"Training took {time.time() - start_time:.2f} seconds")
    
    
    # Step 4: Evaluate Model
    print("Evaluating model...")
    class_names = skin_df['lesion_type'].unique().tolist()
    accuracy, report_df, cm = evaluate_model(model, test_loader, class_names, device, log_dir)
    
    # Step 5: Visualizations
    print("Generating visualizations...")
    show_prediction_examples(model, test_loader, class_names, device, log_dir=log_dir)
    visualize_model_features(model, test_loader, class_names, device, log_dir=log_dir)
    show_gradcam_examples(model, test_loader, class_names, device, log_dir=log_dir)
    
    print(f"Done! All results saved to {log_dir}")

if __name__ == "__main__":
    mp.freeze_support()  # For Windows multiprocessing support
    main()
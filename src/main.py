import torch
import argparse # used for parsing command-line arguments
import torch.multiprocessing as mp
import time
import os
import sys
from datetime import datetime 
import traceback

# from my modules
"""
from data_preparation import prepare_data
from model import create_model, create_criterion, create_optimizer, create_scheduler, device
from train import train_model
from evaluate import evaluate_model, show_prediction_examples, visualize_model_features, show_gradcam_examples
from utils import set_seed """

# Handle imports differently based on how the script is run
try:
    # When running as a module
    from .data_preparation import prepare_data
    from .model import create_model, create_criterion, create_optimizer, create_scheduler, device
    from .train import train_model
    from .evaluate import evaluate_model, show_prediction_examples, visualize_model_features, show_gradcam_examples
    from .utils import set_seed
except ImportError:
    # When running directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_preparation import prepare_data
    from src.model import create_model, create_criterion, create_optimizer, create_scheduler, device
    from src.train import train_model
    from src.evaluate import evaluate_model, show_prediction_examples, visualize_model_features, show_gradcam_examples
    from src.utils import set_seed


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
    parser.add_argument('--pretrained', type=str, default='', help='Path to pre-trained model')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and use pre-trained model')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
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
    
    # Get class names
    class_names = skin_df['lesion_type'].unique().tolist()
    num_classes = len(class_names)
    
    # Step 2: Create Model
    print("Creating model...")
    start_time = time.time()
    model = create_model(num_classes)
    
    # Load pre-trained model if specified
    log_dir = None
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pre-trained model from {args.pretrained}")
        try:
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state dictionary loaded successfully")
            log_dir = os.path.dirname(args.pretrained)
            print(f"Using log directory: {log_dir}")
            if 'accuracy' in checkpoint:
                print(f"Model loaded, validation accuracy: {checkpoint['accuracy']}")
            else:
                print("Model loaded, validation accuracy not found in checkpoint")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
    elif args.pretrained:
        # Only show warning if a path was specified but doesn't exist
        print(f"Warning: Specified pretrained model path '{args.pretrained}' does not exist!")
               
    # Ensure log_dir exists even in evaluation mode
    if log_dir is None or not os.path.exists(log_dir):
        log_dir = os.path.join('logs', 'eval_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created new log directory: {log_dir}")  
        
    criterion = create_criterion()
    optimizer = create_optimizer(model, learning_rate=args.lr)
    scheduler = create_scheduler(optimizer)
    print(f"Model creation took {time.time() - start_time:.2f} seconds")
    
    
    # Step 3: Train Model
    if not args.skip_training and not args.eval_only:
        print(f"Training for {args.num_epochs} epochs...")
        start_time = time.time()
        model, history, log_dir = train_model(
            model, criterion, optimizer, scheduler, 
            dataloaders, dataset_sizes, device,
            num_epochs=args.num_epochs,
            log_dir=log_dir #pass existing log_dir
        )
        print(f"Training took {time.time() - start_time:.2f} seconds")
        
    
    # Step 4: Evaluate Model
    if args.eval_only or not args.skip_training:
        print("Evaluating model...")
        accuracy, report_df, cm = evaluate_model(model, test_loader, class_names, device, log_dir)
    
    # Step 5: Visualizations
    print("Generating visualizations...")
    try:
        show_prediction_examples(model, test_loader, class_names, device, log_dir=log_dir)
        visualize_model_features(model, test_loader, class_names, device, log_dir=log_dir)
        show_gradcam_examples(model, test_loader, class_names, device, log_dir=log_dir)
    except Exception as e:
        print(f"Error during visualization: {e}")
                
        traceback.print_exc()
        
    print(f"Done! All results saved to {log_dir}")

if __name__ == "__main__":
    mp.freeze_support()  # For Windows multiprocessing support
    main()
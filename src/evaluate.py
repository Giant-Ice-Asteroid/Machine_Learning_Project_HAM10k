import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize


##### EVALUATE AND VISUALIZE MODEL PREDICTIONS ###########


### EVALUATE ####

# Evaluation function
def evaluate_model(model, dataloader, class_names, device, log_dir=None):
    """
    
    Arguments:
        model: the trained neural network to be evaluated
        dataloader: provision of batches of test data
        class_names: class names as txt ie melanoma
    """
    
    #model.eval() puts the model into evalation mode: disables dropout layers(=uses all neurons) and uses fixed batch normalization statistics
    model.eval()
    
    #preperaing prediction "collection"
    # -> creates empty list for storing all predictions and true labels
    all_preds = []
    all_labels = []
    
    # Processing test data
    # context manager that disables gradient tracking => not needed and reduces memory usage during eval
    with torch.no_grad():
        
        for inputs, labels in dataloader: # iterates over one batch at a time
            inputs = inputs.to(device) # transfers data to GPU if available
            labels = labels.to(device)
            
            outputs = model(inputs) # forward pass -> the model makes predictions and returns scores for each class
            _, preds = torch.max(outputs, 1) # get predicted class for each image -> takes the index of the highest score as the predicted class (actual score is discarded hence _)
            
            # moves predictions and labels from gpu to cpu with .cpu
            # converts from pytorch tensor to numpy array with .numpy 
            # add the predictions and labels to the lists created earlier
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # calculate accuracy
    # accuracy_score scikit-learn function that calculates proportion of correct predictions
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {accuracy:.4f}")

    #Generate a report 
    # classification_report: scikit-learn function that calculates multiple metrics: 
    # precision (% correct positive preds) -> True Pos / (True pos + False pos) 
    # recall (% correct positives id'ed) -> true pos / (trus pos + false neg)
    # F1: Balances precision and recall into a single score ->  2 * (precision * recall) / (precision + recall)
    # support: Occurrences of each class in the test set
    # target_names=class_names: Uses readable class names in the report (such as Melanoma instead of 1)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose() # .transpose() to flip rows and columns for better display
    print(report_df)
    
    # CONFUSION MATRIX
    # scikit-leanrn function that creates a matrix showing:
    # --> rows: true classes
    # --> columns: predicted classes
    # --> values: count of predictions in each combimnation
    # --> diagonal elements: correct predictions
    # --> off-diagonal elements: incorrect predictions
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues) # displays matrix as a heatmap where darker blue = higher count
    plt.title("Confusion Matrix")
    plt.colorbar() # color scale reference
    # adding class labels:
    tick_marks = np.arange(len(class_names)) # creates evenly spaced positions
    plt.xticks(tick_marks, class_names, rotation=45) # sets tick locations and labels for x axis (rotate for better readability)
    plt.yticks(tick_marks, class_names) # ditto minus rotation
    # adding value annotations:
    thresh = cm.max() / 2. # calculates a threshhold (thresh) for txt color -> half of the max value in the matrix
    for i in range (cm.shape[0]): # i for row index (true class)
        for j in range (cm.shape[1]): # j for column index (predicted class)
            plt.text(j, i, format(cm[i, j], "d"), # adds the count value as txt in each cell, formats to integer 
                        horizontalalignment="center", # centers txt
                        color="white" if cm[i, j] > thresh else "black") # uses white text on dark background and black on light
    plt.tight_layout() # adjusting subplot parameters for better fit
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # save classification report
    if log_dir and os.path.exists(log_dir):
        plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
        report_df.to_csv(os.path.join(log_dir, "classification_report.csv"))
    plt.show()
        
    return accuracy, report_df, cm
  
########################## VISUALISE PREDICTIONS #############

# prediction visualization function
def show_prediction_examples(model, dataloader, class_names, device, num_images=8, log_dir=None):
    """
    Purpose:
    -> Model Verification: Confirms the model is working as expected
    -> Error Analysis: Helps identify patterns in mistakes
    -> Debugging Aid: Reveals potential issues with preprocessing or model design
    model: the trained neural network
    dataloader: the dataloader, for providing batches of test images
    class_names: the classes in text form
    num_images=8: default number of examples to show

    """
    model.eval()
    
    # getting sample images:
    images_so_far = 0
    #lists for correct and incorrect predictions
    correct_preds = []
    incorrect_preds = []
    
    #begnning predicitons process
    with torch.no_grad(): #disablesd gradient calculate (not needed for just making predictions/visulizations)
        for i, (inputs, labels) in enumerate(dataloader):# looping through batches dewlivered by the data loader - enumerate gives the batch index "i"
            print(f"Batch shapes - inputs: {inputs.shape}, labels: {labels.shape}")#Print the shape of inputs and labels for debugging
            inputs = inputs.to(device)
            labels = labels.to(device)  # FIXED: Was incorrectly set to inputs.to(device)
            
            outputs = model(inputs)# model processes the batch, the outputs contains a score for each lesion class
            _, preds = torch.max(outputs, 1)# gets predicted class and takes the highest score index as "preds", actual score discarded (_)
            
            for j in range(inputs.size(0)):#  size(0) to go one img at a time
                if images_so_far >= num_images:
                    break
                
                 # convert tensor to image 
                 # mvoes tensor to cpu, selects the j'th imagae from the batch and converts from pytorcvh tensor to numpy array
                img = inputs[j].cpu().numpy().transpose((1, 2, 0)) # .transpose((1, 2, 0)): rearranges dimensions back (pytorch -> matplotlib)
                 
                # denormalize the image:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean # undo the normalization applied previously
                img = np.clip(img, 0, 1)# ensures all pixel values are between 0 and 1 to avoid visual aritfatcs
                
                # Convert labels and predictions to integers with .item()
                label_idx = labels[j].cpu().item()
                pred_idx = preds[j].cpu().item()
                
                #storing info about each prediction
                pred_info = {
                    "image": img,
                    "true": class_names[label_idx],# actual class name (integer index)
                    "pred": class_names[pred_idx],# predicted class name (int index)
                    "correct": pred_idx == label_idx #bool
                }
                
                if pred_info["correct"]:
                    correct_preds.append(pred_info)
                else:
                    incorrect_preds.append(pred_info)
                
                images_so_far += 1
            
            if images_so_far >= num_images:
                break
    
    # Displaying some correct predictions with visualization
    plt.figure(figsize=(15, 10))
    
    correct_to_show = min(4, len(correct_preds))
    for i in range(correct_to_show):
        plt.subplot(2, 4, i + 1)
        plt.imshow(correct_preds[i]["image"])
        plt.title(f"True: {correct_preds[i]['true']}\nPred: {correct_preds[i]['pred']}")
        plt.axis("off")
    
    for i in range(correct_to_show, 4):
        plt.subplot(2, 4, i + 1)
        plt.axis("off")
    
    incorrect_to_show = min(4, len(incorrect_preds))
    for i in range(incorrect_to_show):
        plt.subplot(2, 4, i + 5)
        plt.imshow(incorrect_preds[i]["image"])
        plt.title(f"True: {incorrect_preds[i]['true']}\nPred: {incorrect_preds[i]['pred']}")
        plt.axis("off")
    
    for i in range(incorrect_to_show, 4):
        plt.subplot(2, 4, i + 5)
        plt.axis("off")
    
    plt.tight_layout()
    
    if log_dir and os.path.exists(log_dir):
        plt.savefig(os.path.join(log_dir, "prediction_examples.png"))
    
    plt.show()
    return correct_preds, incorrect_preds
 
 
############ VISUALISE MODEL FEATURES ###################

def visualize_model_features(model, dataloader, class_names, device, log_dir=None):
    
    """
    Visualization that helps understanding what the model "sees" internally..
    Visualizes feature maps from an intermediate layer of the model
    -> What patterns it has learned to recognize
    -> Identify if it focuses on irrelevant features
    """ 
    model.eval()
    
    try:
        
        # getting a batch of images => changed to single batch for debugging..
        # unpacks the batch into inputs (images) and classes (labels)
        
        inputs, classes = next(iter(dataloader))
        print(f"Visualization batch shapes - inputs: {inputs.shape}, classes: {classes.shape}")
        
        # moves images to gpu if appliacable
        inputs = inputs.to(device)
        
        # Setup feature extraction
        
        # empty list that stores extracted feature maps
        features_blobs = []
        
        # Find appropriate layer to visualize
        if hasattr(model, 'layer4'):
            target_layer = model.layer4
        else:
            # Find the last convolutional layer
            target_layer = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
            
            if target_layer is None:
                print("No suitable convolutional layer found for feature visualization")
                
                # Create an empty visualization as a fallback
                plt.figure(figsize=(15, 12))
                for i in range(15):
                    plt.subplot(3, 5, i + 1)
                    plt.text(0.5, 0.5, "Feature map visualization failed", 
                             horizontalalignment='center', verticalalignment='center')
                    plt.axis("off")
                
                if log_dir and os.path.exists(log_dir):
                    plt.savefig(os.path.join(log_dir, 'feature_maps.png'))
                plt.show()
                return
        
        # create "hook" function that captures layer outputs
        #module = layer being hooked, input = input to this layer, output = output from this layer (to be captured)
        def hook_feature(module, input, output):
            # detach() removes tensor from the computation graph
            # tensor is moved from gpu to cpu if appliacable and converted to numpy array
            # the faeature maps are added to the list above
            features_blobs.append(output.detach().cpu().numpy())

        # registering the hook
        # the hook is registered on a layer of interest (the last convolutional layer)
        # in this case, using ResNEt18, that would be layer4
        # layer4 is the last major convolutional block in ResNet-18
        # layer4 captures high-level features like texture patterns and object parts
        # register_forward_hook(hook_feature) attaches our hook function to this layer
        # -> the hook will then be called whenever data passes through this layer
        hook_handle = model.layer4.register_forward_hook(hook_feature)
        hook_handle = target_layer.register_forward_hook(hook_feature)

        # Forward pass -> making predictions
        # outputs = model(inputs): Runs the images through the model, activating all layers, including layer4
        # hook function captures the feature maps during this process
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        # Remove hook after using it
        hook_handle.remove()
        
        # Check if we got any feature maps

        if not features_blobs:
            print("No feature maps were captured. Check the model architecture.")
            
            # Create an empty visualization as a fallback
            plt.figure(figsize=(15, 12))
            for i in range(15):
                plt.subplot(3, 5, i + 1)
                plt.text(0.5, 0.5, "No feature maps captured", 
                         horizontalalignment='center', verticalalignment='center')
                plt.axis("off")
            
            if log_dir and os.path.exists(log_dir):
                plt.savefig(os.path.join(log_dir, 'feature_maps.png'))
            plt.show()
            return
            
        # accessing capptured feature maps        
        # [0] because only one batch was processed
        # contains feature maps for all images in the batch
        feature_maps = features_blobs[0]
        print(f"Feature maps shape: {feature_maps.shape}")
        
        # visualization
        plt.figure(figsize=(15, 12))
        
        # Process up to 3 images, or fewer if the batch is smaller
        num_images_to_show = min(3, inputs.shape[0])
        
        # iterate through selected images
        # as before convert to numpy and reverse the normalization that was applied during data preparation
        for img_idx in range(num_images_to_show):
            # Get and process the image
            img = inputs[img_idx].cpu().numpy().transpose((1, 2, 0))
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Get the class and prediction for this image
            class_idx = classes[img_idx].cpu().item()
            pred_idx = preds[img_idx].cpu().item()
            
            # Display original image
            plt.subplot(3, 5, img_idx * 5 + 1)
            plt.imshow(img)
            plt.title(f"True: {class_names[class_idx]}\nPred: {class_names[pred_idx]}")
            plt.axis("off")
            
            # Display feature maps for this image (up to 4 maps)
            feature_map = feature_maps[img_idx]
            num_feature_maps = min(4, feature_map.shape[0])
            
            for fmap_idx in range(num_feature_maps):
                plt.subplot(3, 5, img_idx * 5 + fmap_idx + 2)
                plt.imshow(feature_map[fmap_idx], cmap='viridis')
                plt.title(f"Feature Map {fmap_idx}")
                plt.axis('off')
        
        plt.tight_layout()
        
        # Save image if log_dir is provided
        if log_dir and os.path.exists(log_dir):
            plt.savefig(os.path.join(log_dir, 'feature_maps.png'))
        
        plt.show()
        
    except Exception as e:
        print(f"Error in feature visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Create an empty visualization as a fallback
        plt.figure(figsize=(15, 12))
        for i in range(15):
            plt.subplot(3, 5, i + 1)
            plt.text(0.5, 0.5, "Feature visualization failed", 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis("off")
        
        if log_dir and os.path.exists(log_dir):
            plt.savefig(os.path.join(log_dir, 'feature_maps.png'))
        plt.show()
    
 
############ GRAD-CAM VISUALIZATION ############
def generate_gradcam(model, input_tensor, target_layer="layer4"):
    """
    Function for further analysis of medical images
    Grad-CAM = Gradient-weighted Class Activation Mapping
    Show which area of the lesion influenced the model's "diagnosis"
    Confirms whether the model focuses on the lesion itself rather the irrelevant background elements, shows biases
    I.e for melanoma should highloght border and color irregularity, assymmetry etc
    In case of incorrect predictions, can help highlight how it was "misled"
    
    Arguments:
        model: the trained neural network
        input_tensor: the image to be visualized
        taget_layer_name="layer4": layer to be visualized (default is last convolutional layer ie layer4 here)
    """
    
    model.eval()
    
    # temporarily enable gradients for all parameters
    original_grad_states = {}
    for name, param in model.named_parameters():
        original_grad_states[name] = param.requires_grad
        param.requires_grad = True
    
    # access nested layers for ResNet models
    if hasattr(model, 'layer4'):
        target_layer = model.layer4
    elif hasattr(model, 'features') and len(model.features) > 0:
        # Fall back to last convolutional layer if available
        target_layer = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
    else:
        print(f"Cannot find appropriate target layer. Model structure: {model}")
        
        # Restore original gradient states before returning
        for name, param in model.named_parameters():
            param.requires_grad = original_grad_states[name]
        return None, None
    
    # hooks
    
    # create variables for storing captured data
    # initially set as None
    # later used to compute the grad-cam visualization
    feature_maps = None
    gradients = None
    

    # forward hook function
    # captures the output (feature maps) from target layer
    # nonlocal feature_maps: References the outer variable
    # output.detach(): it makes a copy without tracking gradients
    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    # backward hook function
    # captures the gradients flowing back through target layer
    # grad_output[0]: The gradient with respect to the output
    # .detach(): makes a copy without tracking gradients
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        if len(grad_output) > 0:  # Add safety check
            gradients = grad_output[0].detach()
    
    # Register hooks
    #register_forward_hook: attaches our forward hook to the target layer ->  hook runs after the forward pass through this layer
    forward_handle = target_layer.register_forward_hook(forward_hook)
    # register_full_backward_hook: attaches our backward hook to the target layer ->  hook runs during the backwards pass through this layer
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # forward pass
        # run model on the input image -> forward hook captures the feature maps
        output = model(input_tensor)
        
        # get predicted class 
        # argmax(dim=1): returns the index of the maximum value along dimension 1 -> class with high prediction score
        pred_class = output.argmax(dim=1)
        
        #backward pass
        # model.zero_grad(): clears any existing gradients -> only computing gradients for this specific case

        model.zero_grad()
        
        # output[0, pred_class]: accesses the prediction score for the selected class
        # class_score.backward() computes gradients of this score for these model parameters. during this, backward hook captures the gradiens
        class_score = output[0, pred_class]
        class_score.backward()
        
        # Check if hooks captured data
        if feature_maps is None or gradients is None:
            print("Failed to capture feature maps or gradients")
            return None, pred_class
            
         #compute grad-cam visualization
        # averages the gradients across spatial dimensions (heigh and width)
        # -> creates weight for each feature map channel
        # cam = torch.sum(weights * feature_maps..) multiplies each feature map by importance weight and sums across feature channels
        # -> this then creates the class activiation map (CAM) that shows where the model is "looking" when making its decisions
        # cam = torch.relu(cam) applies the ReLU functions that set neg values to zero, i.e CAM focuses only on what positively helps predict
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize CAM
        cam = cam - cam.min()
        if cam.max() > 0:  # Prevent division by zero
            cam = cam / (cam.max() + 1e-7)
        
        return cam, pred_class
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        # Remove hooks
        # important for clean up to avoid memory leak or for hooks to affect future operations
        forward_handle.remove()
        backward_handle.remove()
        
        # Restore original gradient states
        for name, param in model.named_parameters():
            param.requires_grad = original_grad_states[name]


# creating visualizations with the grad cam function just made
def show_gradcam_examples(model, dataloader, class_names, device, num_images=4, log_dir=None):
    """Display Grad-CAM visualizations for multiple images"""
    
    model.eval()
    
    plt.figure(figsize=(15, 10))
    images_processed = 0
    max_attempts = 50  # Limit total attempts to avoid infinite loop
    attempts = 0
    
    try:
        # Process batches until we have enough images or reach max attempts
        for inputs, labels in dataloader:
            if attempts >= max_attempts:
                print(f"Reached maximum attempts ({max_attempts}). Stopping GradCAM visualization.")
                break
                
            attempts += 1
            print(f"GradCAM batch shapes - inputs: {inputs.shape}, labels: {labels.shape}")
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Process each image in the batch
            for j in range(inputs.size(0)):
                if images_processed >= num_images:
                    break
                
                # get single image from the batch
                # j:j+1 slice keeps the batch dimension (required by the model)
                # ->  gives a tensor of the shape (1, channels, height, width)
                img_tensor = inputs[j:j+1]
                
                
                # calls the grad_cam function which returns a heat map and predicted class
                cam, pred_class = generate_gradcam(model, img_tensor)
                
                # Skip if Grad-CAM generation failed
                if cam is None:
                    print(f"Skipping image {j} due to Grad-CAM generation failure")
                    continue
                
                #converting tensors for visualization
                # .squeeze(): Removes singleton dimensions (e.g., from (1, 1, H, W) to (H, W))
                # .transpose((1, 2, 0)): Rearranges from (C,H,W) to (H,W,C) format
                cam = cam.cpu().squeeze().numpy()
                img = img_tensor.cpu().squeeze().numpy().transpose((1, 2, 0))
                
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # resize CAM to match image size and ensure proper overlay of heatmap on image
                # img.shape[0]: Image height
                # img.shape[1]: Image width
                cam = resize(cam, (img.shape[0], img.shape[1]))               
                
                # Convert tensor indices to integers
                label_idx = labels[j].cpu().item()
                pred_idx = pred_class.item()
                
                # Create visualization        
                plt.subplot(2, 2, images_processed + 1)
                plt.imshow(img) #display the original image
                #overlay grad-cam as heatmap
                # cmap='jet': Uses a colormap ranging from blue (low) to red (high)
                # alpha=0.5: Sets transparency (50% transparent)
                plt.imshow(cam, cmap="jet", alpha=0.5) 
                
                true_label = class_names[label_idx]
                pred_label = class_names[pred_idx] 
                plt.title(f"True: {true_label}\nPred: {pred_label}")
                plt.axis("off")
                
                images_processed += 1
            
            if images_processed >= num_images:
                break 
                
        # Handle case where we didn't find enough images
        if images_processed == 0:
            print("No images were processed for Grad-CAM. Creating empty visualization.")
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.text(0.5, 0.5, "GradCAM failed", 
                         horizontalalignment='center', verticalalignment='center')
                plt.axis("off")
        
        plt.tight_layout()
    
        if log_dir and os.path.exists(log_dir):     
            plt.savefig(os.path.join(log_dir, "gradcam_examples.png"))
        
        plt.show()
        
    except Exception as e:
        print(f"Error in Grad-CAM visualization: {e}")
        import traceback
        traceback.print_exc()

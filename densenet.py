import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.mps
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from torchvision.models import densenet121
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################

# Part 2: Predefined Model
class DenseNet_CIFAR100(nn.Module):
    def __init__(self, num_classes=100):
        super(DenseNet_CIFAR100, self).__init__()
        
        # Load DenseNet-121 model
        self.densenet = densenet121(weights=None)  # No pre-trained weights

        # Modify the first convolution layer to adapt for CIFAR-100 (32x32 images)
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove the first max pooling layer (not needed for small images)
        self.densenet.features.pool0 = nn.Identity()

        # Modify the classifier for CIFAR-100 (100 classes instead of 1000)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(inputs)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        ### TODO
        # Add loss from this batch
        running_loss += loss.item()  

        ### TODO
        _, predicted = torch.max(outputs, 1)  

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total, "lr": optimizer.param_groups[0]["lr"]})

    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)
            
            ### TODO -- inference
            outputs = model(inputs) 
            
            ### TODO -- loss calculation
            loss = criterion(outputs, labels)    

            ### SOLUTION -- add loss from this sample
            # Add loss from this batch
            running_loss += loss.item() 

            ### SOLUTION -- predict the class
            _, predicted = torch.max(outputs, 1)     

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": "DenseNet",     # Change name when using a different model
        "batch_size": 64,        # run batch size finder to find optimal batch size
        "learning_rate": 0.1,    # Learning rate for SGD
        "momentum": 0.9,         # Momentum for SGD
        "weight_decay": 5e-4,    # L2 penalty
        "epochs": 20,             # Train for longer in a real scenario
        "num_workers": 8,        # Adjust based on your system
        "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        "data_dir": "./data",    # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                   # Random crop 32x32 with padding 4 black pixels
        transforms.RandomHorizontalFlip(p=0.5),                 # Randomly flip the image horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),  # Best normalization for CIFAR-100
    ])

    ###############
    # TODO Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    ### TODO -- BEGIN SOLUTION
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),  # Best normalization for CIFAR-100
    ])

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    
    ### TODO -- Calculate training set size
    train_size = int(0.8 * len(trainset))  

    ### TODO -- Calculate validation set size
    val_size = len(trainset) - train_size 

    ### TODO -- split into training and validation sets
    trainset, valset = random_split(trainset, [train_size, val_size])  

    ### TODO -- define loaders and test set
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"],
                            shuffle=True, num_workers=CONFIG["num_workers"])
    
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=CONFIG["num_workers"])

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                            download=True, transform=transform_test)
    
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"],   
                            shuffle=False, num_workers=CONFIG["num_workers"])
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
      
    # instantiate your model ### TODO
    model = DenseNet_CIFAR100() 

    # move it to target device
    model = model.to(CONFIG["device"])

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    
    ### TODO -- define loss criterion
    criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for classification   

    ### TODO -- define optimizer
    # weight delay for L2 penalty 
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])  

    # Add a scheduler   
    # ### TODO -- you can optionally add a LR scheduler

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR by a factor of 0.1 every 5 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])  
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)  # Reduce LR when validation loss plateaus

    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    # Check if the pre-trained model exists
    model_path = "best_model.pth"
    start_epoch = 0  # Default to 0 if no model is found

    if os.path.exists(model_path):
        # Load the model weights
        checkpoint = torch.load(model_path, map_location=CONFIG["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # Load optimizer state
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # Load scheduler state (if any)
        start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
        best_val_acc = checkpoint["best_val_acc"]  # Use the stored best accuracy
    else:
        print(f"No pre-trained model found at {model_path}. Training from scratch.")  
        best_val_acc = 0.0

    print("\nModel summary:")
    print(f"{model}\n")

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    
    # best_val_acc = 0.0

    for epoch in range(start_epoch, CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        
        # Update the learning rate scheduler
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,                                     # Current epoch
                "model_state_dict": model.state_dict(),             # Model weights
                "optimizer_state_dict": optimizer.state_dict(),     # Optimizer state
                "scheduler_state_dict": scheduler.state_dict(),     # Scheduler state
                "best_val_acc": best_val_acc                        # Best validation accuracy
            }, model_path)

            wandb.save(model_path) # Save to wandb as well

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()

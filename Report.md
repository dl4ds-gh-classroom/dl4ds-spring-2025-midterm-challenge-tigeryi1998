# DS542 Midterm Report 

### Sicheng Yi (Tiger Yi)
### Kaggle ID: tigeryi98
### BU ID: U43188754
### Email: tigeryi@bu.edu

    * Report Requirement
    * Comprehensive AI disclosure statement.
    * Completeness and clarity of the report.
    * Thoroughness of experiment tracking.
    * Justification of design choices.
    * Analysis of results.
    * Ablation study (if included).


## AI disclosure statement

In this midterm project, I have used Github Copilot and ChatGPT to guide me through hyperparameters tuning, model layers changes, as well as the pros of cons of various learning rate scheduler and optimizers. I will go into details on where I used generative AIs to guide me.

1. Learning rate scheduler. I was using StepLR to have fixed steps LR scheduler. However, the model training isn't take into the fact that learning rate should be decaying if the model is trained for some time. ChatGPT gave me options of CosineLR and ReduceLROnPlateau. I tried to use both. 

2. Optimizer. I was using Adam and AdamW originally. I search in ChatGPT in various ways to improve training. It told me that SGD works better in CNN and Adam is more used in transformers. Therefore I switch to the SGD optimizer instead. 

3. Model modifications. I used ChatGPT as a recommendation to slightly change the 3 model structures. 
For the SimpleCNN I designed, it asks me to use techniques like dropout and batch norm. I was using 2 conv and 2 fc layers and ChatGPT asks me to add more layers so I changed to 3 layers. For predefined DenseNet training from scratch, ChatGPT helped me to change the first conv0 layer because DenseNet was trained on larger ImageNet 224x224 images, not the 32x32 images in CIFAR100. For the pretrained and finetuned DenseNet, ChatGPT helped me to freeze the first few dense layers and unfreeze the rest so I can keep some pretrained weights to fine tune. 

4. Early Stopping. The most time consuming part is to figure out how many epochs to train and how to retrain or keep training from the previous models. ChatGPT really helped in the early stopping so I am not wasting epochs and GPU time if the model is not improving much in validation. It saves me many compute time as well. 

Overall, ChatGPT and Github Copilot really helped and guided me in some of the common practices and explains what is the pros and cons of different models and tools. I personally code up most of the code myself but the guidance of AI really saved me time to do extra experiments. 


## Files

https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-tigeryi1998

Python files

```bash
simplecnn.py
densenet.py
finetune.py
```
Bash files

```bash
simplecnn.sh
densenet.sh
finetune.sh
```

Submission_ood.csv is too big to upload to Github

output files
```bash
[simplecnn, densenet, finetune].o#######
```

## Data Transformation 

For CIFAR100 32x32 Images. First split into 80% of training images and 20% validation images. The rest of test imags won't ever go into model training or validation and test imags will be held out. 

For both the training set images (train + validation) images and the test set images. All images are normalized by the training set mean and standard deviation. 

mean=(0.5071, 0.4867, 0.4408)
std=(0.2675, 0.2565, 0.2761)

All images have the same normalization to ensure it's all the same for model during training, validation, and the test evalution. 

It's also a good normalization technique using mean and standard deviation of the actual data rather than the default 0.5 normalization before. 

Only the training images will have Random Horizontal Flip of 50% chance. This is to make sure our model can learn the same object just by inverting the images. Flip the image should not change the label classification. 

Also only for training images, Random Crop is applied. First pad 4 black pixels around the 32x32 image into 36x36, then it will randomly crop out the 32x32 section to make new images. This will help the model to be more robust and be position invariant. If object just move away it is still the same object. 

I did not use other augmentation like Verticle Flip and Reducing Lighting because many objects can't be upside down and change in lighting might impact the object detection (debatable point). I don't have time to experiment other augmentation anyway. 


## Models 

### SimpleCNN

Model summary:

```
SimpleCNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=2048, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=100, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
)
```

For my SimpleCNN models. I have 3 convolutional layers, 3 batch norm layers, 3 max pool layers, 1 Flatten layer, 2 Fully connected layers with drop out in between. Input images is Batch x 3 x 32 x 32. So I used 3 convo layers to make it to B x 32 x 32 x 32, B x 64 x 32 x 32, B x 128 x 32 x 32. 

The 3 conv layers will have batch norm and relu, and max pool to downsize hidden layers. After everything will be flatten and feed into 3 fully connected layers with a 50% drop out in between. In the end the final layer will have 100 classes and output logits. 


### DenseNet121, predefined train from scratch 

Model summary:

```
DenseNet_CIFAR100
  (densenet): DenseNet
    (features): Sequential
      (conv0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu0): ReLU(inplace=True)
      (pool0): Identity()
      (denseblock1): _DenseBlock
      ...
      (denseblock4)
      ...
      (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (classifier): Linear(in_features=1024, out_features=100, bias=True)
```

DenseNet121 is the predefined model I choose to use to train from the scratch with no pretrained weights. Compare to other torchvision models like VGG or ResNet, DenseNet is fast to train because fewer parameters in the fully connected layers and it works well. 

Importantly, I change the inner layers of Conv0 to kernal size 3, stride 1. Used to be kernal size 7, stride 2. This is because model was trained on Imagenet 224x224 but CIFAR100 is smaller 32x32 images so I don't want it to downsize early. 

self.densenet.features.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

I also change the first maxpool layer to be identity because again I don't want to downsize images at the start. model was trained on Imagenet 224x224 but CIFAR100 is smaller 32x32 images. 

self.densenet.features.pool0 = nn.Identity()


### DenseNet121, pretrained weights and finetune

Same model structure than before but with pretrained weights load in. I try to finetune this model with existing weights by freezing the first 2 densenet blocks and unfreeze the rest later layers. 

        # Freeze all initial layers
        for param in self.densenet.features[0:3].parameters():
            param.requires_grad = False 

        # Unfreeze the dense blocks
        for param in self.densenet.features[3:].parameters():
            param.requires_grad = True 

        # Unfreeze the final classifier layer to fine-tune it
        for param in self.densenet.classifier.parameters():
            param.requires_grad = True

## Training

### optimizer: SGD
I use SGD because ChatGPT says SGD is better than Adam in CNN while another is more used in transformer. 
For SGD, I tried different hyperparamters like learning rates lr=0.1 or lr=0.05. lower learning rate will be slower to converge but not overshoot. For lower learning rate, I need more epochs. 

### loss criterion: CrossEntropyLoss
Cross Entropy Loss is good for multi class classification 

### learning rate scheduler: StepLR, CosineAnnealingLR, ReduceLROnPlateau
I asked ChatGPT on common types of lr schedulers. First I tried StepLR on SimpleCNN and DenseNet, it doesn't count for the progress of training and learning rate should decrease overtime. Then, I also tried CosineAnnealingLR which is good for long epochs of traning. But I think learning rate decay to fast at each epochs. I eventually settle with ReduceLROnPlateau, it at least takes into account of validation loss and accuracy. So no learning rate reduction until val_loss stops improving much then we take smaller steps. 

### CONFIG

batch_size=64, I mostly kept the batch size of 64 in most models in the end as I tune it I think it's big enough so it won't train too slow and small enough it won't eat up all the CPUs. 

learning_rate = 0.01, 0.005, I tried different learning rate to see which training will be better. It seems fine tuned model needs lower learning rate than training from scratch. 

epochs = 10, 20, 50, 75 I tried a few different epochs during experiments. It's very slow to train 50 epochs or more so I tried to see if 10-20 epochs give a good progress. I also used early stopping so in the end 75 epochs are not needed and it stops around 50 epochs. 

weight_decay = 5e-4, 1e-4, 1e-3. SGD weight decay I tried different L2 regularization factors to prevent overfitting. Larger weight_decay might be slower but preventation overfitting is worth as the model is training only on CIFAR100 training set. It is better to generalize the models for unknown test data. 


To run everything on SCC, I created 3 bash script to run. In the terminal, type:

```bash
qsub simplecnn.sh

qsub densenet.sh

qsub finetune.sh
```


## Results

### SimpleCNN

```
wandb: Run history:
wandb:      epoch ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb:         lr ████▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁
wandb:  train_acc █▁▃▃▂▄▂▂▂▃▇▄▃▅▅█████
wandb: train_loss ▇████▄▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:    val_acc ▇▄██▇▁▂▁▁▂▂▂▂▁▁▁▁▁▁▁
wandb:   val_loss ▆█▅▃▆▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:      epoch 20
wandb:         lr 1e-05
wandb:  train_acc 1.055
wandb: train_loss 4.60499
wandb:    val_acc 0.78
wandb:   val_loss 4.60709

Evaluating on Clean Test Set:   0%|          | 0/313 [00:00<?, ?it/s]
Evaluating on Clean Test Set:   0%|          | 1/313 [00:00<00:41,  7.45it/s]
Evaluating on Clean Test Set:  18%|█▊        | 56/313 [00:00<00:00, 286.89it/s]
Evaluating on Clean Test Set:  36%|███▌      | 113/313 [00:00<00:00, 404.92it/s]
Evaluating on Clean Test Set:  54%|█████▍    | 169/313 [00:00<00:00, 462.01it/s]
Evaluating on Clean Test Set:  73%|███████▎  | 227/313 [00:00<00:00, 500.47it/s]
Evaluating on Clean Test Set:  91%|█████████ | 284/313 [00:00<00:00, 520.47it/s]
Evaluating on Clean Test Set: 100%|██████████| 313/313 [00:00<00:00, 438.44it/s]
Clean CIFAR-100 Test Accuracy: 1.00%
All files are already downloaded.
```

SimpleCNN after training 20 epochs only has accuracy around 1% on the test set, 1% on train image, less 1% on validation. 

The model has too few convolutional and fully connected layers. It is on par with benchmark on Kaggle. 



### DenseNet121, predefined train from scratch 

```
wandb: uploading densenet2.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:      epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:         lr ██████████████████▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:  train_acc ▁▂▃▃▄▄▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇███████████████
wandb: train_loss █▇▆▅▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:    val_acc ▁▂▃▄▄▅▅▅▅▆▆▅▆▆▆▆▆▆▆████████████████████
wandb:   val_loss █▇▅▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:      epoch 39
wandb:         lr 1e-05
wandb:  train_acc 95.16
wandb: train_loss 0.19428
wandb:    val_acc 74.68
wandb:   val_loss 0.91858
wandb: 
wandb: 🚀 View run fiery-eon-19 at: https://wandb.ai/tigeryi-boston-university/-sp25-ds542-challenge/runs/nxgrxm0g
wandb: ⭐️ View project at: https://wandb.ai/tigeryi-boston-university/-sp25-ds542-challenge
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20250330_131533-nxgrxm0g/logs
Early stopping triggered!
Early stopping at epoch 39
...
Evaluating on Clean Test Set: 100%|██████████| 157/157 [00:07<00:00, 22.17it/s]
Evaluating on Clean Test Set: 100%|██████████| 157/157 [00:07<00:00, 20.43it/s]
Clean CIFAR-100 Test Accuracy: 75.22%
All files are already downloaded.
```

DenseNet after training 75 epochs from the scratch with no pretrained weights.

The model early stops at 39 epochs as validation loss and accuracy has not improved by much. 

It has 75% accuracy on the CIFAR test images, 74% on the valdiation images, 95% on the training images

This is my most accurate model out of the 3 models. 


### DenseNet121, pretrained weights and finetune

```
wandb: uploading finetune.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:      epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:         lr █████████████████████████▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:  train_acc ▁▂▃▃▄▅▅▅▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆▆▆▇▇▇▇███████████
wandb: train_loss █▆▆▅▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:    val_acc ▁▂▃▄▄▅▅▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆███████████████
wandb:   val_loss █▇▅▅▅▄▃▃▃▃▃▃▃▃▃▃▃▃▃▂▃▂▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:      epoch 50
wandb:         lr 5e-05
wandb:  train_acc 93.5025
wandb: train_loss 0.25605
wandb:    val_acc 72.96
wandb:   val_loss 0.97302
wandb: 
wandb: 🚀 View run lilac-sponge-20 at: https://wandb.ai/tigeryi-boston-university/-sp25-ds542-challenge/runs/5bwqkli7
wandb: ⭐️ View project at: https://wandb.ai/tigeryi-boston-university/-sp25-ds542-challenge
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20250330_150543-5bwqkli7/logs
Early stopping triggered!
Early stopping at epoch 50
...
Evaluating on Clean Test Set:  98%|█████████▊| 154/157 [00:04<00:00, 40.69it/s]
Evaluating on Clean Test Set: 100%|██████████| 157/157 [00:04<00:00, 37.92it/s]
Clean CIFAR-100 Test Accuracy: 74.21%
All files are already downloaded.
```

Trained all 50 epochs with no early stopping. 

The DenseNet model with pretrained weights from ImageNet has 74% accuracy on the test images, 73% accuracy on the validation images, and 93% accuracy on the traning images. 

This model is slightly worse than the DenseNet just trained from scratch. But it could be that it needs more epochs as early stopping did not trigger. 

Overall the accuracy is comparable to the previous model. 


### result summary

For the SimpleCNN, the accuracy is extremely bad as the model is not sophisticated enough to learn. 

For both the predefined and pretrain-finetune models, the train accuracy is over 90%,
but the valdiation and test accuracy is around 75%. 

I think it could be that both models overfit on the train images and didn't fully generalize to validation and test images. But early stoping, weight decay L2 regularization in the optimization should mitigate the overfitting problem 


## Conclusion 

In this midterm challenge, I have implement many CNN and DL techniques, such as: 

convolutional, dropoout, batchnorm, flatten, relu, fully connected layers

early stopping, hyperparameter tuning, L2 regularization

data augmentation such as horizontal flip, mean standard deviation normalization 

Overall the 3 models achieve decent results and beat out the baseline metrics on Kaggle. 
















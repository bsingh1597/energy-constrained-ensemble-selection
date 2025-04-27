
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from initializeModel import initializeModel
from trainModel import trainModel

print("Inititiating code execution")

# TODO Define data directories
data_dir = '/home/myid/bs83243/mastersProject/imagenet_dataset/'

import torchvision.transforms as transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  # More aggressive cropping range
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30, fill=(124, 116, 104)),  # Moderate rotation, fill with mean color (approx. ImageNet mean in RGB)
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Color jitter
        # transforms.RandomGrayscale(p=0.2), # Optional: Random grayscale conversion
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), # Resize larger first
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Create datasets
image_datasets = {
    'train': datasets.ImageNet(root=data_dir, split='train', transform=data_transforms['train']),
    'val': datasets.ImageNet(root=data_dir, split='val', transform=data_transforms['val']),
}
batchSize = 32
# Create dataloaders for train and validation
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batchSize, shuffle=True, num_workers=12, pin_memory=True) for x in ['train', 'val']}
print(f"Data loader batch size {batchSize}")
# TODO Define models to train
# models_to_train = ['resnext50', 'resnet152', 'resnet18', 'densenet201', 'efficientnet_b0', 'squeezeNet1_1', 'vgg19_bn', 'alexnet', 'vgg16', 'inception_v3']
models_to_train = ['squeezeNet1_1']

# squeezeNet1_1 , batch size- 1024, lr-3.2  - failing
# Alexnet failing
# vgg16 failing


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
print("Device:", device)

# Train each model
for model_name in models_to_train:
    print(f"Training {model_name}...")
    print("="*50)
     
    # Initialize the model
    model = initializeModel(model_name)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,  # Much lower learning rate
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True)
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    print(f"Scheduler parameters: {scheduler.state_dict()}")

    # Train the model
    trained_model = trainModel(device, model, model_name, image_datasets, dataloaders, criterion, optimizer, scheduler, num_epochs=90)

    # Save the model
    torch.save({
        'model_state_dict':trained_model.state_dict(),
        }, f'{model_name}_imagenet.pth')
    
    print(f"{model_name} training complete.\n")

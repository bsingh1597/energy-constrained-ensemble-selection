import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

def initializeModel(model_name, num_classes=1000, pretrained=True, weights= None):
    """Initialize a model with random weights based on the name"""
    if model_name == "resnext50":
        model = models.resnext50_32x4d(models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    elif model_name == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    elif model_name == 'squeezeNet1_1':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
    elif model_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
    if model_name in ['resnext50', 'resnet152', 'resnet18']:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(model.fc.bias)
    elif model_name == 'densenet201':
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        nn.init.kaiming_normal_(model.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(model.classifier.bias)
    elif model_name == 'efficientnet_b0':
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, num_classes),)
    elif model_name == 'inception_v3':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
        # Initialize the bias to zeros
        nn.init.zeros_(model.fc.bias)
        
        # Handle auxiliary classifier if enabled
        if model.AuxLogits is not None:
            num_ftrs_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
            # Initialize auxiliary classifier weights
            nn.init.kaiming_normal_(model.AuxLogits.fc.weight, 
                                mode='fan_out', 
                                nonlinearity='relu')
            nn.init.zeros_(model.AuxLogits.fc.bias)
    elif model_name == 'squeezeNet1_1':
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif model_name in ['vgg19_bn', 'vgg16']:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        # Initialize the weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(model.classifier[6].weight)
        # Initialize the bias to zeros
        nn.init.zeros_(model.classifier[6].bias)
        # Ensure dropout is set correctly (0.5 is the standard for both networks)
        if hasattr(model.classifier, '2'):  # Dropout after first FC layer
            model.classifier[2].p = 0.5
        if hasattr(model.classifier, '5'):  # Dropout after second FC layer
            model.classifier[5].p = 0.5
    elif model_name == 'alexnet':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    return model



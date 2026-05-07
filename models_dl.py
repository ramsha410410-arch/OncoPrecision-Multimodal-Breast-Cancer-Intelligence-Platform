import torch
import torch.nn as nn
from torchvision import models, transforms

def get_transfer_learning_model(num_classes=2):
    """
    Implements ResNet18 Transfer Learning.
    Freezes base layers and replaces the head for medical classification.
    """
    # Load pre-trained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Image Transformation pipeline for the UI
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
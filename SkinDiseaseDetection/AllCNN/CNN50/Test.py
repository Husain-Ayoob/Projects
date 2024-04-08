# Importing necessary libraries
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test dataset
dataset_path = "Dataset/SkinDisease/test"
dataset = ImageFolder(root=dataset_path, transform=transform)

# Data loader for the test set
test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

model = resnet50(weights=None)  
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5), 
    torch.nn.Linear(num_ftrs, 19)  
)
model_load_path = "SkinModel.pth"
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#Training Loss: 0.0040, Training Accuracy: 99.89%, Validation Loss: 0.0012, Validation Accuracy: 99.98%
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset_path = "VEG"
dataset = ImageFolder(root=dataset_path, transform=transform)

train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

# data loaders for the training and validation sets
train_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(val_indices))
print("initialize model")

model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  
print("Loss Function")

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training and validation loop
num_epochs = 3
print("num epochs: ", num_epochs)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc="Training", unit="batch"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_running_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Save the model
model_save_path = "model\\model101.pth"
torch.save(model.state_dict(), model_save_path)

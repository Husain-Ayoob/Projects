import torch
from torchvision import transforms
from torchvision.models import resnet101
from PIL import Image
import os
from torchvision.transforms import functional as TF
"""
Class 0: OverRipe
Class 1: Ripe
Class 2: Rotten
Class 3: UnRipe

"""
dataset_path = "Test Images"
model_load_path = "model/model101.pth"
results_file_path = "predictions.txt" 


transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')), 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  

model.load_state_dict(torch.load(model_load_path))

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

def predict_and_log(model, image_paths, transform, results_file_path):
    with open(results_file_path, 'w') as results_file:
        for image_path in image_paths:
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0)
            
            if torch.cuda.is_available():
                image = image.cuda()
            
            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                predicted_class = preds.item()
            
            # Write the image name and its predicted class to the text file
            results_file.write(f"{os.path.basename(image_path)}: {predicted_class}\n")

predict_and_log(model, image_paths, transform, results_file_path)

print("Image classification complete. Results saved to predictions.txt.")

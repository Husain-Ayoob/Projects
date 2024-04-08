import torch
from torchvision import transforms
from torchvision.models import resnet101
from PIL import Image
import cv2
import numpy as np
import os

# Define classes for readability
CLASS_LABELS = ["OverRipe", "Ripe", "Rotten", "UnRipe"]

def setup_model(model_load_path):
    model = resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image, transform):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = preds.item()
    return predicted_class

def detect_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bruise_range = [np.array([10, 50, 50]), np.array([30, 255, 255])]
    unripe_range = [np.array([30, 50, 50]), np.array([70, 255, 255])]
    unripe_mask = cv2.inRange(hsv_image, unripe_range[0], unripe_range[1])
    bruise_mask = cv2.inRange(hsv_image, bruise_range[0], bruise_range[1])
    not_unripe_mask = cv2.bitwise_not(unripe_mask)
    bruise_mask = cv2.bitwise_and(bruise_mask, not_unripe_mask)
    kernel = np.ones((5, 5), np.uint8)
    bruise_mask = cv2.morphologyEx(bruise_mask, cv2.MORPH_OPEN, kernel)
    return bruise_mask

def calculate_area(mask):
    return cv2.countNonZero(mask)

def process_predictions(model, images, transform):
    result = []
    for image in images:
        predicted_class = predict_image(model, image, transform)
        if predicted_class in [0, 2]:  # OverRipe or Rotten
            open_cv_image = np.array(image) 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            bruise_mask = detect_features(open_cv_image)
            bruise_area = calculate_area(bruise_mask)
            total_area = open_cv_image.shape[0] * open_cv_image.shape[1]
            bruise_percentage = (bruise_area / total_area) * 100
            result.append((CLASS_LABELS[predicted_class], bruise_percentage))
        else:
            result.append((CLASS_LABELS[predicted_class], 0))  # Ripe or UnRipe with 0% bruise
    return result

def rottenCNN(images):
    model_load_path = "CNNs/model101.pth"
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = setup_model(model_load_path)

    results = process_predictions(model, images, transform)
    return results

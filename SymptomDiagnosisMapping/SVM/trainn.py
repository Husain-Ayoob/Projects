import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os


df = pd.read_csv('Dataset/Symptoms/augmented_dataset.csv', encoding='utf-8')
df.fillna(0, inplace=True)

print("Data loaded.")

labels = df['label']
data = df.drop('label', axis=1)

print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

C = 1.0
gamma = 'scale'

# Train SVM model
svm_model = SVC(C=C, gamma=gamma)
svm_model.fit(X_train, y_train)

# Evaluate SVM model
predictions = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

print("Model evaluation metrics:")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


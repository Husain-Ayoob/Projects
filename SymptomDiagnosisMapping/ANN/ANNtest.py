import pandas as pd
import numpy as np
from keras.models import load_model

model = load_model('ANN.keras')
classes = np.loadtxt('label_classes.npy', delimiter=',', dtype=str)

try:
    test_df = pd.read_csv('Dataset/Symptoms/Medical_Test.csv', encoding='utf-8')
except UnicodeDecodeError:
    test_df = pd.read_csv('Dataset/Symptoms/Medical_Test.csv', encoding='ISO-8859-1')
test_df.fillna(0, inplace=True)

predictions = model.predict(test_df)

for i, prediction in enumerate(predictions):
    top_5_indices = prediction.argsort()[-5:][::-1]
    top_5_probabilities = prediction[top_5_indices]
    
    # Ensure indices are within the bounds of the classes array
    top_5_indices = [index for index in top_5_indices if index < len(classes)]
    top_5_classes = classes[top_5_indices]

    print(f"Test Instance {i+1}:")
    for j, cls in enumerate(top_5_classes):
        print(f"{j+1}. {cls} (Probability: {top_5_probabilities[j]*100:.2f}%)")
    print("\n")

import pandas as pd
import numpy as np

try:
    df = pd.read_csv('Dataset/Symptoms/medical_conditions.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('Dataset/Symptoms/medical_conditions.csv', encoding='ISO-8859-1')
df.fillna(0, inplace=True)

print("Data loaded")

def toggle_symptoms(row):
    new_row = row.copy()
    symptom_indices = list(range(len(row) - 1))
    num_symptoms_to_toggle = np.random.randint(1, 3)
    symptoms_to_toggle = np.random.choice(symptom_indices, size=num_symptoms_to_toggle, replace=False)
    for idx in symptoms_to_toggle:
        new_row[idx] = 1 - new_row[idx]
    return new_row

augmented_data = pd.DataFrame(columns=df.columns)
np.random.seed(42)
counter = 0

while len(augmented_data) < 100000:
    for _, row in df.iterrows():
        augmented_row = toggle_symptoms(row.values[:-1]).tolist()
        augmented_row.append(row.values[-1])
        new_row_df = pd.DataFrame([augmented_row], columns=df.columns)
        augmented_data = pd.concat([augmented_data, new_row_df], ignore_index=True)
        counter = counter + 1
        print(counter)
        if len(augmented_data) >= 100000:
            break

augmented_data.to_csv('augmented_dataset.csv', index=False)
print("Augmentation complete. Dataset saved.")

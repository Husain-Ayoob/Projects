import pandas as pd
from joblib import load
import pickle
# Loading the model
with open('random_forest_model.pkl', 'rb') as file:
    clf = pickle.load(file)

try:
    df = pd.read_csv('Dataset\\Symptoms\\medical_conditions.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('Dataset\\Symptoms\\medical_conditions.csv', encoding='ISO-8859-1') 

df.fillna(0, inplace=True)




predictions = clf.predict(df)
print("Predictions:", predictions)

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Load the model
model_filename = 'multioutput_clf.joblib'
multioutput_clf = joblib.load(model_filename)

mlb_filename = 'mlb.joblib'  
mlb = joblib.load(mlb_filename)

try:
    df = pd.read_csv('Dataset\\Symptoms\\Medical_Test.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('Dataset\\Symptoms\\Medical_Test.csv', encoding='ISO-8859-1')
df.fillna(0, inplace=True)

X_pred = df

# Predict on the data
y_pred = multioutput_clf.predict(X_pred)

predicted_labels = mlb.inverse_transform(y_pred)

# Output the predicted diagnoses names
for i, labels in enumerate(predicted_labels):
    print(f"Case {i+1}: {', '.join(labels) if labels else 'No diagnosis found'}")

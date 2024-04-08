import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('Dataset/Symptoms/augmented_dataset.csv', encoding='utf-8')

df.fillna(0, inplace=True)

print("loaded data")
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training")
# Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("evaluation")
# Evaluation
predictions = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("saving")
import pickle

# Saving the model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro')  # Adjust 'average' as needed
recall = recall_score(y_test, predictions, average='macro')  # Adjust 'average' as needed
f1 = f1_score(y_test, predictions, average='macro')  # Adjust 'average' as needed

print("Model evaluation metrics:")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
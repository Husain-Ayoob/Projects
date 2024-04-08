import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from joblib import dump

df = pd.read_csv('Dataset/Symptoms/augmented_dataset.csv', encoding='utf-8')
df.fillna(0, inplace=True)
df['label'] = df['label'].apply(lambda x: x.split(','))
print("1")
# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['label'])

X = df.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_clf = lgb.LGBMClassifier()

# Wrap LightGBM model to handle multi-label tasks
multioutput_clf = MultiOutputClassifier(lgb_clf, n_jobs=-1)

# Training
multioutput_clf.fit(X_train, y_train)

y_pred = multioutput_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

f1 = f1_score(y_test, y_pred, average='macro')
print(f'F1 Score: {f1}')

recall = recall_score(y_test, y_pred, average='macro')
print(f'Recall: {recall}')

precision = precision_score(y_test, y_pred, average='macro')
print(f'Precision: {precision}')

model_filename = 'multioutput_clf.joblib'
dump(multioutput_clf, model_filename)

mlb_filename = 'mlb.joblib'
dump(mlb, mlb_filename)

print(f'Model saved to {model_filename}')
print(f'MultiLabelBinarizer saved to {mlb_filename}')

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical  
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

for encoding in ['utf-8', 'ISO-8859-1']:
    try:
        train_df = pd.read_csv('Dataset/Symptoms/augmented_dataset.csv', encoding=encoding)
        break
    except UnicodeDecodeError:
        continue

train_df.fillna(0, inplace=True)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_dummy = to_categorical(y_train_encoded)

model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(y_train_dummy.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_dummy, epochs=15, batch_size=15, verbose=1)
model.save('ANN.keras')

predictions_dummy = model.predict(X_train)
predictions = np.argmax(predictions_dummy, axis=1) 

accuracy = accuracy_score(y_train_encoded, predictions)
precision = precision_score(y_train_encoded, predictions, average='macro')
recall = recall_score(y_train_encoded, predictions, average='macro')
f1 = f1_score(y_train_encoded, predictions, average='macro')

print("Model evaluation metrics:")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

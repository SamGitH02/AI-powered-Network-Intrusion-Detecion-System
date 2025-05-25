import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load the new dataset
file_path = 'CIC Dataset/cybersecurity_intrusion_data.csv'
df = pd.read_csv(file_path)

# Drop non-numerical columns that are not useful for training
df_clean = df.drop(['session_id', 'protocol_type', 'encryption_used', 'browser_type'], axis=1)

# Debugging: Print the columns after preprocessing
print("Columns after preprocessing:", df_clean.columns.tolist())

# Handle missing values
df_clean = df_clean.fillna(0)

# Encode the target column ('attack_detected')
label_encoder = LabelEncoder()
df_clean['attack_detected'] = label_encoder.fit_transform(df_clean['attack_detected'])

# Separate features and labels
X = df_clean.drop('attack_detected', axis=1)
y = df_clean['attack_detected']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_samples = X_scaled.shape[0]
num_features = X_scaled.shape[1]

# Update the input shape to match the actual number of features
height = 1
width = 6

if num_features != height * width:
    raise ValueError(f"The number of features ({num_features}) does not match the specified height ({height}) and width ({width}).")

X_reshaped = X_scaled.reshape(num_samples, height, width, 1)

y_categorical = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(height, width, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

try:
    # Experiment with a smaller batch size for training
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, callbacks=[lr_scheduler])

    # Save the retrained model
    model.save('model_trained_on_new_dataset.h5')
    print("Model trained and saved as 'model_trained_on_new_dataset.h5'")

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_acc:.4f}')

    predictions = model.predict(X_test)
except Exception as e:
    print(f"An error occurred: {str(e)}")

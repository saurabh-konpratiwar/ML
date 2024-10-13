import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

def neuralNetwork_Model():
    # Load your data
    data = pd.read_csv('MLApi/data/train.csv')  # Updated path to train.csv

    # Prepare your features and labels
    X = data.drop(columns=['fake'])  # Replace 'fake' with your target column name
    y = data['fake']
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize your data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the neural network model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # For binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

    # Save the model
    model.save('MLApi/data/neural_model.h5')
    print("Model trained and saved successfully!")

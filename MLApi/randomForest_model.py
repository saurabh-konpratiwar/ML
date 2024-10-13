import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def trainRandomForest_model():
    # Load your dataset
    df = pd.read_csv('MLApi/data/train.csv')  # Updated path to train.csv

    # Preprocessing: define your features and target
    X = df.drop('fake', axis=1)  # All columns except the target
    y = df['fake']  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'MLApi/data/random_forest_model.pkl')

    print("Model trained and saved successfully!")


import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from the text file
data_file = "data.txt"
data = np.loadtxt(data_file)

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Labels are the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

with open('./model', 'wb') as f:
    pickle.dump(rf_classifier, f)


# If you want to load the model later:
# with open('emotion_recognition_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, 
#     classification_report, 
#     confusion_matrix
# )
# import joblib

# def load_and_prepare_data(file_path):
#     """
#     Load data from a text or CSV file and prepare it for model training.
    
#     Parameters:
#     -----------
#     file_path : str
#         Path to the data file containing features
    
#     Returns:
#     --------
#     X : numpy array
#         Feature matrix
#     y : numpy array
#         Target labels
#     """
#     # Load the data
#     try:
#         # Try reading as CSV first
#         data = pd.read_csv(file_path, header=None)
#     except Exception:
#         # If CSV fails, try reading as text file
#         try:
#             data = pd.read_csv(file_path, sep='\s+', header=None)
#         except Exception as e:
#             print(f"Error reading file: {e}")
#             raise

#     # Assume the last column is the target variable, others are features
#     X = data.iloc[:, :-1].values
#     y = data.iloc[:, -1].values

#     return X, y

# def train_random_forest(X, y, n_estimators=100, test_size=0.2, random_state=42):
#     """
#     Train a Random Forest Classifier
    
#     Parameters:
#     -----------
#     X : numpy array
#         Feature matrix
#     y : numpy array
#         Target labels
#     n_estimators : int, optional (default=100)
#         Number of trees in the forest
#     test_size : float, optional (default=0.2)
#         Proportion of the dataset to include in the test split
#     random_state : int, optional (default=42)
#         Controls the randomness of the training
    
#     Returns:
#     --------
#     dict containing model, scaler, and evaluation metrics
#     """
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )

#     # Scale the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Train Random Forest Classifier
#     rf_classifier = RandomForestClassifier(
#         n_estimators=n_estimators, 
#         random_state=random_state, 
#         n_jobs=-1  # Use all available cores
#     )
#     rf_classifier.fit(X_train_scaled, y_train)

#     # Make predictions
#     y_pred = rf_classifier.predict(X_test_scaled)

#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     class_report = classification_report(y_test, y_pred)

#     # Print evaluation metrics
#     print("Model Evaluation:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print("\nConfusion Matrix:")
#     print(conf_matrix)
#     print("\nClassification Report:")
#     print(class_report)

#     # Feature importance
#     feature_importance = rf_classifier.feature_importances_
#     print("\nFeature Importance:")
#     for i, importance in enumerate(feature_importance):
#         print(f"Feature {i+1}: {importance:.4f}")

#     # Save the model and scaler
#     joblib.dump(rf_classifier, 'random_forest_model.joblib')
#     joblib.dump(scaler, 'feature_scaler.joblib')

#     return {
#         'model': rf_classifier,
#         'scaler': scaler,
#         'accuracy': accuracy,
#         'confusion_matrix': conf_matrix,
#         'classification_report': class_report
#     }

# def main():
#     # Path to your data file
#     data_path = r"D:\Uppsala Stuff\P2\Intelligent\finalproject\data.txt"  # Update this with your actual file path
    
#     # Load data
#     X, y = load_and_prepare_data(data_path)
    
#     # Train the model
#     results = train_random_forest(X, y)

# if __name__ == '__main__':
#     main()

# # Example of how to use the saved model for predictions
# def predict_new_data(new_data_path):
#     """
#     Load saved model and scaler to make predictions on new data
    
#     Parameters:
#     -----------
#     new_data_path : str
#         Path to new data file for prediction
#     """
#     # Load saved model and scaler
#     model = joblib.load('random_forest_model.joblib')
#     scaler = joblib.load('feature_scaler.joblib')
    
#     # Load new data
#     new_data = pd.read_csv(new_data_path, header=None).values
    
#     # Scale the new data
#     new_data_scaled = scaler.transform(new_data)
    
#     # Make predictions
#     predictions = model.predict(new_data_scaled)
    
#     return predictions

# # Note: Ensure you have the following libraries installed:
# # pip install numpy pandas scikit-learn joblib


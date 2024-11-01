# credit_churnc_predection_codsoft
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('your_data.csv')  # Replace with your dataset path

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop the 'Surname' column as it is not useful for analysis
data.drop(columns=['Surname'], inplace=True)

# Convert categorical variables into numerical ones
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Define features and target variable
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Function to predict churn probability for all users
def predict_churn_probabilities(data):
    # Drop the 'Exited' column to prepare for predictions
    X_data = data.drop('Exited', axis=1)

    # Align with training data (fill missing columns with 0)
    for col in X.columns:
        if col not in X_data.columns:
            X_data[col] = 0

    # Ensure the order of columns matches the training data
    X_data = X_data[X.columns]

    # Scale the features
    X_scaled = scaler.transform(X_data)

    # Predict probabilities
    churn_probabilities = rf_model.predict_proba(X_scaled)

    # Add churn probabilities to the original DataFrame
    data['Churn Probability (%)'] = churn_probabilities[:, 1] * 100  # Convert to percentage

    return data

# Get churn probabilities for all users
data_with_churn_probabilities = predict_churn_probabilities(data)

# Display the results
print(data_with_churn_probabilities[['CustomerId', 'Churn Probability (%)']])


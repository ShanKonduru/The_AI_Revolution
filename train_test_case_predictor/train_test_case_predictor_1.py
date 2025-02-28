
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def train_test_case_predictor(data_path, model_path):
    """Trains a machine learning model to predict test case pass/fail."""

    # Load data
    data = pd.read_csv(data_path)

    # Preprocessing
    le_environment = LabelEncoder()
    le_input = LabelEncoder()
    le_expected = LabelEncoder()
    le_actual = LabelEncoder()

    data['Environment'] = le_environment.fit_transform(data['Environment'])
    data['InputData'] = le_input.fit_transform(data['InputData'].astype(str)) #convert to string due to possible null values
    data['ExpectedResult'] = le_expected.fit_transform(data['ExpectedResult'].astype(str))
    data['ActualResult'] = le_actual.fit_transform(data['ActualResult'].astype(str))
    data['PassOrFail'] = data['PassOrFail'].map({'Pass': 1, 'Fail': 0})

    # Features and target
    features = ['Environment', 'InputData', 'ExpectedResult', 'ActualResult']
    target = 'PassOrFail'

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Save model and label encoders
    joblib.dump({
        'model': model,
        'le_environment': le_environment,
        'le_input': le_input,
        'le_expected': le_expected,
        'le_actual': le_actual
    }, model_path)

    print(f"Model saved to {model_path}")

# Example usage:
model_path =  "result\\train_test_case_predictor_1.joblib"
train_test_case_predictor("input\\train_test_case_predictor_0.csv", model_path) #replace test_data.csv with your file name.

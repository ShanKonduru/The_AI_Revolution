
import pandas as pd
import joblib

def predict_test_case_result(model_path, environment, input_data, expected_result, actual_result):
    """Predicts the pass/fail result of a test case."""

    # Load model and encoders
    data = joblib.load(model_path)
    model = data['model']
    le_environment = data['le_environment']
    le_input = data['le_input']
    le_expected = data['le_expected']
    le_actual = data['le_actual']

    # Preprocess input
    environment_encoded = le_environment.transform([environment])[0]
    input_encoded = le_input.transform([str(input_data)])[0] #convert to string
    expected_encoded = le_expected.transform([expected_result])[0]
    actual_encoded = le_actual.transform([actual_result])[0]

    # Predict
    prediction = model.predict([[environment_encoded, input_encoded, expected_encoded, actual_encoded]])

    # Interpret prediction
    if prediction[0] == 1:
        return "Pass"
    else:
        return "Fail"

# Example usage:
model_path_list = {
    "result\\train_test_case_predictor_1.joblib", 
    "result\\train_test_case_predictor_2.joblib", 
    "result\\train_test_case_predictor_3.joblib", 
    } 

for model_path in model_path_list:
    environment = "Chrome"
    input_data = "user=test"
    expected_result = "Login Successful"
    actual_result = "Login Successful"

    result = predict_test_case_result(model_path, environment, input_data, expected_result, actual_result)
    print(f"Model: {model_path} Predicted Result: {result}")

    environment = "Edge"
    input_data = "quantity=1000"
    expected_result = "Item added"
    actual_result = "Error message"

    result = predict_test_case_result(model_path, environment, input_data, expected_result, actual_result)
    print(f"Model: {model_path} Predicted Result: {result}")

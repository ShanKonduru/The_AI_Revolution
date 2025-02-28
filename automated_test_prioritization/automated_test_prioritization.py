import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def automated_test_prioritization(data_path, model_path="test_priority_model.joblib"):
    """Prioritizes test cases based on predicted defect density using clustering and classification."""

    # Load data
    data = pd.read_csv(data_path)

    # Preprocessing
    le_environment = LabelEncoder()
    le_input = LabelEncoder()
    le_expected = LabelEncoder()
    le_actual = LabelEncoder()

    data['Environment'] = le_environment.fit_transform(data['Environment'])
    data['InputData'] = le_input.fit_transform(data['InputData'].astype(str))
    data['ExpectedResult'] = le_expected.fit_transform(data['ExpectedResult'].astype(str))
    data['ActualResult'] = le_actual.fit_transform(data['ActualResult'].astype(str))
    data['PassOrFail'] = data['PassOrFail'].map({'Pass': 1, 'Fail': 0})

    # Features for clustering (including PassOrFail patterns)
    cluster_features = ['Environment', 'InputData', 'ExpectedResult', 'ActualResult', 'PassOrFail']

    # Clustering (K-Means)
    kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
    data['Cluster'] = kmeans.fit_predict(data[cluster_features])

    # Calculate defect density per cluster
    cluster_defect_density = data.groupby('Cluster')['PassOrFail'].mean() #Fail is 0, pass is 1, so mean represents the fail ratio.
    data['DefectDensity'] = data['Cluster'].map(cluster_defect_density)
    print("Cluster Defect Density:")
    print(cluster_defect_density)

    # Bin DefectDensity into discrete classes
    bins = [0, 0.33, 0.66, 1.0]  # Adjust bins as needed
    labels = ['Low', 'Medium', 'High']
    data['DefectDensityClass'] = pd.cut(data['DefectDensity'], bins=bins, labels=labels, include_lowest=True)
    
    print("\nDefect Density Class Counts:")
    print(data['DefectDensityClass'].value_counts())

    # Classification (Predicting Defect Density Class)
    X = data[cluster_features]
    y = data['DefectDensityClass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Defect Density Class Prediction Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nPredicted Defect Density Classes (Test):")
    print(y_pred)

    # Prioritization (Rank by Predicted Defect Density Class)
    data['PredictedDefectDensityClass'] = model.predict(data[cluster_features])
    prioritized_tests = data.sort_values(by='PredictedDefectDensityClass', ascending=False)

    # Visualization (Cluster Distribution and Defect Density)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.countplot(x='Cluster', data=data)
    plt.title('Test Case Cluster Distribution')

    plt.subplot(1, 2, 2)
    sns.barplot(x='Cluster', y='DefectDensity', data=data)
    plt.title('Defect Density per Cluster')
    plt.tight_layout()
    # plt.show()

    # Save the model and relevant data
    joblib.dump({
        'model': model,
        'le_environment': le_environment,
        'le_input': le_input,
        'le_expected': le_expected,
        'le_actual': le_actual,
        'kmeans': kmeans,
        'cluster_defect_density': cluster_defect_density
    }, model_path)

    return prioritized_tests

# Example Usage
model_path = "result\\automated_test_prioritization.joblib"
prioritized_tests = automated_test_prioritization("input\\automated_test_prioritization.csv", model_path)

print("Prioritized Test Cases:")
print(prioritized_tests[['TestCaseID', 'PredictedDefectDensityClass']])
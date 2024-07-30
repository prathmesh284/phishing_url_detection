import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from custom_transformers import FeatureExtractor

# Load dataset
data = pd.read_csv('Dataset/phishing_site_urls.csv')

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('feature_extractor', FeatureExtractor())
])

# Full pipeline with Random Forest Classifier
model_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Create feature matrix
X = data['URL']
# Convert categorical labels to numerical labels (0 and 1)
y = data['Label'].map({'bad': 0, 'good': 1}).values  # Map 'bad' to 0 and 'good' to 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Evaluate model on training set
y_train_pred = model_pipeline.predict(X_train)
print('Training Set Performance:')
print(f'Accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f'Precision: {precision_score(y_train, y_train_pred)}')
print(f'Recall: {recall_score(y_train, y_train_pred)}')
print(f'F1 Score: {f1_score(y_train, y_train_pred)}')

# Evaluate model on test set
y_test_pred = model_pipeline.predict(X_test)
print('Test Set Performance:')
print(f'Accuracy: {accuracy_score(y_test, y_test_pred)}')
print(f'Precision: {precision_score(y_test, y_test_pred)}')
print(f'Recall: {recall_score(y_test, y_test_pred)}')
print(f'F1 Score: {f1_score(y_test, y_test_pred)}')

# Save the model
joblib.dump(model_pipeline, 'phishing_model.pkl')

def predict_url(url):
    features = FeatureExtractor().transform([url])
    prediction = model_pipeline.predict(features)[0]
    result = 'phishing' if prediction == 0 else 'not phishing'
    return result

# Predict on a new URL
url = "http://www.google.com"
result = predict_url(url)
print(f'The URL {url} is {result}')
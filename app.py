from flask import Flask, request, render_template,jsonify
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
import joblib
from custom_transformers import FeatureExtractor

# Load the trained model
model = joblib.load('phishing_model.pkl')

# Create Flask app
app = Flask(__name__)

# Endpoint for predicting URL
@app.route('/predict', methods=['POST'])
def predict_url():
    data = request.get_json(force=True)
    url = data.get('url', '')
    features = FeatureExtractor().transform([url])

    prediction = model.predict(features)[0]
    if ('https' in url):
        prediction = 1
    else:
        prediction = 0
    result = 'looking suspisious,be aware of it.' if prediction == 0 else 'Safe.'
    return jsonify({'url': url, 'prediction': result})

# Endpoint to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/phishing')
def phishing():
    return render_template('phishing.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import joblib
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load resources
print("Loading model, label encoder, and dataset...")
model = load_model('disease_classification_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
df = pd.read_csv('pathology_tests_dataset.csv')

def classify_image(image_path):
    """
    Classifies the image into a disease category.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image. Could not read file.", None
    
    # Preprocess image
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_encoder.classes_[predicted_class_index]
    
    # Normalize prediction
    original_predicted_class = predicted_class
    if "photos" in predicted_class.lower():
        final_predicted_class = predicted_class.lower().replace("photos", "").strip().capitalize()
    else:
        final_predicted_class = predicted_class
    
    return final_predicted_class, original_predicted_class

def get_pathology_test(disease_name):
    """
    Returns the recommended pathology test for the detected disease.
    """
    match = df[df.iloc[:, 0].str.strip().str.lower() == disease_name.lower().strip()]
    if not match.empty:
        return match.iloc[0, 1]
    else:
        return "No matching pathology test found."

# Routes
@app.route('/')
def index():
    """
    Renders the home page.
    """
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """
    Handles image classification and pathology test recommendation.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
    
    # Save uploaded image
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Classify image
    predicted_disease, original_predicted_disease = classify_image(file_path)
    
    if not original_predicted_disease:
        return jsonify({"error": predicted_disease}), 500  # Send classification error
    
    # Get pathology test recommendation
    pathology_test = get_pathology_test(original_predicted_disease)
    
    # Clean up uploaded file
    os.remove(file_path)
    
    # Response
    response = {
        "predicted_disease": predicted_disease,
        "pathology_test": pathology_test
    }
    return jsonify(response)

# Main entry point
if __name__ == '__main__':
    # Ensure 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Run Flask app
    app.run(debug=True)

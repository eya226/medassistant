import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory

# --- Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Configuration ---
MODEL_PATH = '../saved_model/best_model.keras'
IMG_SIZE = 128
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(os.path.join(app.static_folder, UPLOAD_FOLDER)):
    os.makedirs(os.path.join(app.static_folder, UPLOAD_FOLDER))

# --- Load Model ---
# Load the trained model once when the app starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Preprocessing Function ---
def preprocess_image(image_stream):
    """
    Reads an image from a stream, preprocesses it, and prepares it for the model.
    """
    try:
        # Convert the file stream to a numpy array
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        # Decode the numpy array as a grayscale image
        gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if gray_img is None:
            return None

        # Resize the grayscale image
        resized_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))

        # Convert grayscale to 3-channel BGR
        bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

        # Normalize and reshape for the model
        bgr_img = bgr_img / 255.0
        bgr_img = np.reshape(bgr_img, (1, IMG_SIZE, IMG_SIZE, 3))

        return bgr_img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Routes ---
@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Preprocess the image
        processed_image = preprocess_image(file)

        if processed_image is None:
            return jsonify({'error': 'Invalid image file or preprocessing failed'}), 400

        # Make prediction
        prediction_prob = model.predict(processed_image)[0][0]
        prediction_class = 'Pneumonia' if prediction_prob > 0.5 else 'Normal'
        confidence = float(prediction_prob) if prediction_class == 'Pneumonia' else 1.0 - float(prediction_prob)

        # Return the result
        return jsonify({
            'prediction': prediction_class,
            'confidence': f'{confidence:.2%}' # Format as percentage
        })

    return jsonify({'error': 'An unknown error occurred'}), 500

# This allows serving uploaded files if needed, e.g., to display on the results page
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.static_folder, UPLOAD_FOLDER), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

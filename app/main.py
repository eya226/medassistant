import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, url_for
import uuid
import json

# --- Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Configuration ---
MODEL_DIR = 'saved_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'brain_tumor_model.keras')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'brain_tumor_class_indices.json')
IMG_SIZE = 150
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, UPLOAD_FOLDER)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model and Class Indices ---
model = None
class_indices = {}
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_INDICES_PATH) as f:
            class_indices = json.load(f)
        print("Brain tumor model and class indices loaded successfully.")
    else:
        print("Warning: Brain tumor model or class indices file not found.")
except Exception as e:
    print(f"Error loading model or class indices: {e}")

# --- Image Processing ---
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_for_model = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0

    return img_for_model, img # Return original BGR image for saving

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected.'}), 400

    results = []
    for file in files:
        image_bytes = file.read()
        processed_image, original_image = preprocess_image(image_bytes)

        if processed_image is None:
            results.append({'filename': file.filename, 'error': 'Invalid image file'})
            continue

        preds = model.predict(np.expand_dims(processed_image, axis=0))[0]
        pred_index = np.argmax(preds)
        pred_class = class_indices.get(str(pred_index), "Unknown")
        confidence = preds[pred_index]

        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the original uploaded image to be displayed with the results
        cv2.imwrite(output_path, original_image)

        analysis_text = ""
        if pred_class != "notumor":
            analysis_text = f"The model predicts a <strong>{pred_class}</strong> with <strong>{confidence:.2%}</strong> confidence. This case should be prioritized for review."
        else:
            analysis_text = f"The model predicts <strong>No Tumor</strong> with <strong>{confidence:.2%}</strong> confidence."

        results.append({
            'filename': file.filename,
            'prediction': pred_class,
            'confidence': f'{confidence:.2%}',
            'image_url': url_for('static', filename=f'{UPLOAD_FOLDER}/{filename}'),
            'analysis_text': analysis_text
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

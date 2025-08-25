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
MODEL_PATH = os.path.join(MODEL_DIR, 'best_multiclass_model.keras')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
IMG_SIZE = 128
UPLOAD_FOLDER = 'uploads'
LAST_CONV_LAYER_NAME = "relu"
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, UPLOAD_FOLDER)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model and Class Indices ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    print("Model and class indices loaded successfully.")
except Exception as e:
    print(f"Error loading model or class indices: {e}")
    model = None
    class_indices = {}

# --- Grad-CAM and Image Processing Functions ---
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, None
    original_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_for_model = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) / 255.0
    img_for_model = np.expand_dims(img_for_model, axis=0)
    return img_for_model, original_img

def generate_grad_cam(model, img_array, last_conv_layer_name, class_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return superimposed_img

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected for uploading'}), 400

    results = []
    for file in files:
        image_bytes = file.read()
        processed_image, original_image = preprocess_image(image_bytes)

        if processed_image is None:
            results.append({'filename': file.filename, 'error': 'Invalid image file'})
            continue

        preds = model.predict(processed_image)[0]
        pred_index = np.argmax(preds)
        pred_class = class_indices.get(str(pred_index), "Unknown")
        confidence = preds[pred_index]

        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        analysis_text = ""

        if pred_class != "Normal":
            heatmap = generate_grad_cam(model, processed_image, LAST_CONV_LAYER_NAME, pred_index)
            superimposed_img = overlay_heatmap(original_image, heatmap)
            cv2.imwrite(output_path, superimposed_img)
            analysis_text = f"The model predicts <strong>{pred_class}</strong> with <strong>{confidence:.2%}</strong> confidence. The heatmap highlights the area of concern."
        else:
            cv2.imwrite(output_path, original_image)
            analysis_text = f"The model predicts <strong>Normal</strong> with <strong>{confidence:.2%}</strong> confidence. No significant visual markers for disease were detected."

        results.append({
            'filename': file.filename,
            'prediction': pred_class,
            'confidence': f'{confidence:.2%}',
            'image_url': url_for('static', filename=f'{UPLOAD_FOLDER}/{filename}', _external=True),
            'analysis_text': analysis_text
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

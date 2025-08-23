import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import uuid

# --- Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Configuration ---
MODEL_PATH = 'saved_model/best_model.keras'
IMG_SIZE = 128
UPLOAD_FOLDER = 'uploads'
LAST_CONV_LAYER_NAME = "relu" # Last convolutional layer name in DenseNet121
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, UPLOAD_FOLDER)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Grad-CAM and Image Processing Functions ---

def preprocess_image(image_bytes):
    """Prepares image bytes for the model."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    # Keep original for overlay
    original_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Preprocess for model
    img_for_model = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) # Model expects RGB
    img_for_model = img_for_model / 255.0
    img_for_model = np.expand_dims(img_for_model, axis=0)

    return img_for_model, original_img

def generate_grad_cam(model, img_array, last_conv_layer_name):
    """Generates a Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlays a heatmap on an image."""
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    superimposed_img = heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# --- Routes ---
@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload, prediction, and analysis."""
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        image_bytes = file.read()
        processed_image, original_image = preprocess_image(image_bytes)

        if processed_image is None:
            return jsonify({'error': 'Invalid image file or preprocessing failed'}), 400

        # Make prediction
        prediction_prob = model.predict(processed_image)[0][0]
        prediction_class = 'Pneumonia' if prediction_prob > 0.5 else 'Normal'
        confidence = float(prediction_prob) if prediction_class == 'Pneumonia' else 1.0 - float(prediction_prob)

        # Generate unique filename for the output image
        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        analysis_text = ""

        if prediction_class == 'Pneumonia':
            # Generate and overlay heatmap
            heatmap = generate_grad_cam(model, processed_image, LAST_CONV_LAYER_NAME)
            superimposed_img = overlay_heatmap(original_image, heatmap)
            cv2.imwrite(output_path, superimposed_img)

            # Generate written analysis
            analysis_text = (
                f"The model predicts <strong>Pneumonia</strong> with <strong>{confidence:.2%}</strong> confidence. "
                "The highlighted region on the X-ray indicates the primary area of concern identified by the AI. "
                "This analysis is AI-generated and should be verified by a qualified medical professional."
            )
        else:
            # Save the original image
            cv2.imwrite(output_path, original_image)
            analysis_text = (
                f"The model predicts <strong>Normal</strong> with <strong>{confidence:.2%}</strong> confidence. "
                "No significant visual markers for pneumonia were detected by the AI. "
                "This analysis is AI-generated and should be verified by a qualified medical professional."
            )

        # Return the result
        return jsonify({
            'prediction': prediction_class,
            'confidence': f'{confidence:.2%}',
            'image_url': url_for('static', filename=f'{UPLOAD_FOLDER}/{filename}'),
            'analysis_text': analysis_text
        })

    return jsonify({'error': 'An unknown error occurred'}), 500

# This is now handled by the 'image_url' in the response, but kept for safety
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, url_for
import uuid
import json
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

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

# --- Image Processing and LIME Functions ---
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, None
    # LIME works best with the original image size before resizing for the model
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Image for the model needs to be resized
    img_for_model = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    img_for_model_norm = img_for_model / 255.0
    return img_for_model_norm, original_img

def get_prediction_function(model):
    """Wrapper function to get model predictions in the format LIME expects."""
    def predict_fn(images):
        return model.predict(images)
    return predict_fn

def generate_lime_explanation(image_for_model, original_image, model, num_features=5):
    """Generates a LIME explanation image."""
    explainer = lime_image.LimeImageExplainer()
    prediction_fn = get_prediction_function(model)

    explanation = explainer.explain_instance(
        image_for_model,
        prediction_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000 # Number of perturbed images to generate
    )

    # Get the explanation for the top class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )

    # Resize the mask to the original image size for overlay
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Mark boundaries on the original image
    explained_image = mark_boundaries(original_image, mask_resized)
    explained_image = (explained_image * 255).astype(np.uint8)
    # Convert back to BGR for saving with OpenCV
    explained_image_bgr = cv2.cvtColor(explained_image, cv2.COLOR_RGB2BGR)

    return explained_image_bgr

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

        # Model expects a batch, so add a dimension
        preds = model.predict(np.expand_dims(processed_image, axis=0))[0]
        pred_index = np.argmax(preds)
        pred_class = class_indices.get(str(pred_index), "Unknown")
        confidence = preds[pred_index]

        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        analysis_text = ""

        if pred_class != "notumor":
            explained_image = generate_lime_explanation(processed_image, original_image, model)
            cv2.imwrite(output_path, explained_image)
            analysis_text = f"The model predicts a <strong>{pred_class}</strong> with <strong>{confidence:.2%}</strong> confidence. The highlighted areas are the most influential regions for this prediction."
        else:
            # For "no tumor", we still need to save the original image to display it
            cv2.imwrite(output_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
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

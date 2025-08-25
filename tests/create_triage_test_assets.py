import numpy as np
import cv2
import os
import json

def create_dummy_image(path, size=(128, 128)):
    """Creates a simple grayscale dummy image."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.randu(image, 0, 255)
    cv2.imwrite(path, image)

def create_dummy_class_indices(path):
    """Creates a dummy class indices JSON file."""
    indices = {
        "0": "Atelectasis",
        "1": "Edema",
        "2": "Normal",
        "3": "Pneumonia",
        "4": "Pneumothorax",
        "5": "Tuberculosis"
    }
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(indices, f)

if __name__ == '__main__':
    # Create assets for testing
    print("Creating test assets...")
    # Create a few dummy images
    for i in range(3):
        create_dummy_image(f'tests/dummy_image_{i}.png')

    # Create dummy model and class files for the loading test
    os.makedirs('saved_model', exist_ok=True)
    create_dummy_class_indices('saved_model/class_indices.json')
    with open('saved_model/best_multiclass_model.keras', 'w') as f:
        f.write('dummy model file')

    print("Test assets created.")

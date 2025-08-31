import numpy as np
import cv2
import os
import json

def create_dummy_image(path, size=(150, 150)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.imwrite(path, image)

def create_dummy_class_indices(path):
    indices = { "0": "glioma", "1": "meningioma", "2": "notumor", "3": "pituitary" }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(indices, f)

if __name__ == '__main__':
    print("Creating test assets...")
    for i in range(3):
        create_dummy_image(f'tests/dummy_image_{i}.png')

    os.makedirs('saved_model', exist_ok=True)
    create_dummy_class_indices('saved_model/brain_tumor_class_indices.json')
    with open('saved_model/brain_tumor_model.keras', 'w') as f:
        f.write('dummy model file')
    print("Test assets created.")

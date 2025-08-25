import pytest
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock
from app.main import app as flask_app

# --- Test Setup and Fixtures ---

@pytest.fixture(scope='module')
def test_assets():
    """Create all necessary dummy files for the test module."""
    print("\nSetting up test assets...")
    # Create dummy images
    image_paths = []
    for i in range(3):
        path = f'tests/dummy_image_{i}.png'
        image_paths.append(path)
        # Use a helper to create the image
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.putText(img, f"Img {i}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(path, img)

    yield {'image_paths': image_paths}

    # Teardown: clean up files
    print("\nTearing down test assets...")
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

@pytest.fixture
def client():
    """Create a test client with mocked model and class indices."""
    flask_app.config['TESTING'] = True

    # Mock the model and its predict method
    mock_model = MagicMock()
    # Simulate a multi-class output for 6 classes
    dummy_prediction = np.array([0.1, 0.05, 0.05, 0.7, 0.05, 0.05]) # Predicts class 3 (Pneumonia)
    mock_model.predict.return_value = [dummy_prediction]

    # Mock the class indices file
    dummy_indices = { "0": "Atelectasis", "1": "Edema", "2": "Normal", "3": "Pneumonia", "4": "Pneumothorax", "5": "Tuberculosis" }

    # Patch the loaders in the app's context
    with patch('app.main.tf.keras.models.load_model', return_value=mock_model):
        with patch('builtins.open', new_callable=MagicMock) as mock_open:
            # When the json file is opened, make it return our dummy indices
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(dummy_indices)

            from app import main
            importlib.reload(main) # Reload app to apply mocks

            with flask_app.test_client() as test_client:
                yield test_client

# Need to import these here because they are used in the fixtures
import cv2
import importlib


# --- Test Cases ---

def test_home_page(client):
    """Test if the home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"AI TriageScan" in response.data

def test_predict_no_files(client):
    """Test the /predict endpoint with no files."""
    response = client.post('/predict', data={}, content_type='multipart/form-data')
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'No files selected' in json_data['error']

def test_predict_multiple_files_success(client, test_assets):
    """Test a successful prediction with multiple image files."""
    files_to_upload = []
    for path in test_assets['image_paths']:
        files_to_upload.append(('files[]', (open(path, 'rb'), os.path.basename(path))))

    response = client.post('/predict', data=dict(files_to_upload), content_type='multipart/form-data')

    # Close the opened files
    for _, (file_handle, _) in files_to_upload:
        file_handle.close()

    assert response.status_code == 200
    json_data = response.get_json()

    # Check if we got results for all 3 images
    assert isinstance(json_data, list)
    assert len(json_data) == 3

    # Check the structure of the first result
    first_result = json_data[0]
    assert 'filename' in first_result
    assert 'prediction' in first_result
    assert 'confidence' in first_result
    assert 'image_url' in first_result
    assert 'analysis_text' in first_result

    # Check the prediction based on our mock
    assert first_result['prediction'] == 'Pneumonia'
    assert first_result['confidence'] == '70.00%'

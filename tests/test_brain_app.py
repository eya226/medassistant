import pytest
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock
import cv2
import importlib
from app.main import app as flask_app

@pytest.fixture(scope='module')
def test_assets():
    """Create dummy image files."""
    image_paths = []
    for i in range(3):
        path = f'tests/dummy_image_{i}.png'
        image_paths.append(path)
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        cv2.imwrite(path, img)
    yield {'image_paths': image_paths}
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

@pytest.fixture
def client():
    """Create a test client with mocked model and class indices."""
    flask_app.config['TESTING'] = True

    mock_model = MagicMock()
    # Predicts class 0 (glioma)
    dummy_prediction = np.array([0.8, 0.1, 0.05, 0.05])
    mock_model.predict.return_value = [dummy_prediction]

    dummy_indices = { "0": "glioma", "1": "meningioma", "2": "notumor", "3": "pituitary" }

    with patch('app.main.tf.keras.models.load_model', return_value=mock_model):
        with patch('builtins.open') as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(dummy_indices)
            from app import main
            importlib.reload(main)
            with flask_app.test_client() as test_client:
                yield test_client

def test_home_page(client):
    """Test the home page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Brain Tumor Analyzer" in response.data

@patch('app.main.generate_grad_cam')
def test_predict_multiple_files(mock_grad_cam, client, test_assets):
    """Test successful prediction with multiple files."""
    mock_grad_cam.return_value = np.random.rand(150, 150)

    files_to_upload = [('files[]', (open(p, 'rb'), os.path.basename(p))) for p in test_assets['image_paths']]
    response = client.post('/predict', data=dict(files_to_upload), content_type='multipart/form-data')
    for _, (handle, _) in files_to_upload:
        handle.close()

    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)
    assert len(json_data) == 3

    first_result = json_data[0]
    assert first_result['prediction'] == 'glioma'
    assert first_result['confidence'] == '80.00%'

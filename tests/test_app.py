import pytest
import os
from unittest.mock import patch, MagicMock
from app.main import app as flask_app

@pytest.fixture
def client():
    """
    Create a test client for the Flask app, with model loading mocked.
    """
    # The 'with' statement ensures the patch is active during the client's lifetime
    with patch('app.main.tf.keras.models.load_model') as mock_load_model:
        # Configure the mock to simulate a successfully loaded model
        mock_model = MagicMock()
        # Set a default prediction. Tests can override this if needed.
        mock_model.predict.return_value = [[0.9]]  # Simulates 'Pneumonia'
        mock_load_model.return_value = mock_model

        # The app loads the model at startup. Since we are patching after
        # the app object has been imported, we need to manually
        # set the model on the imported app's context.
        from app import main
        main.model = mock_model

        flask_app.config['TESTING'] = True
        with flask_app.test_client() as test_client:
            yield test_client # This is what the tests will receive as 'client'

def test_predict_success(client):
    """Test successful prediction with a dummy image."""
    image_path = 'tests/dummy_image.png'
    assert os.path.exists(image_path), "Dummy image not found. Run create_test_assets.py"

    with open(image_path, 'rb') as img:
        data = {'file': (img, 'dummy_image.png')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['prediction'] == 'Pneumonia'
    assert json_data['confidence'] == '90.00%'

def test_home_page(client):
    """Test if the home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Analyze X-Ray Image" in response.data

def test_predict_no_file(client):
    """Test prediction endpoint with no file."""
    response = client.post('/predict', data={}, content_type='multipart/form-data')
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'No file part' in json_data['error']

def test_predict_invalid_file_type(client):
    """Test prediction with a non-image file."""
    text_file_path = 'tests/dummy_file.txt'
    assert os.path.exists(text_file_path), "Dummy text file not found. Run create_test_assets.py"

    with open(text_file_path, 'rb') as txt:
        data = {'file': (txt, 'dummy_file.txt')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')

    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'Invalid image file' in json_data['error']

# A setup function to ensure assets exist before tests run
def setup_module(module):
    """Create dummy files needed for testing."""
    from .create_test_assets import create_dummy_image, create_dummy_text_file
    create_dummy_image()
    create_dummy_text_file()

# Cleanup the dummy files after tests are done
def teardown_module(module):
    """Remove dummy files created for testing."""
    image_path = 'tests/dummy_image.png'
    text_path = 'tests/dummy_file.txt'
    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists(text_path):
        os.remove(text_path)
    print("\nCleaned up test assets.")

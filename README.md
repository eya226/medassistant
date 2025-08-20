# Deep Learning Medical Image Classifier

This project is a full-stack deep learning application for classifying chest X-ray images to detect pneumonia. It includes Python scripts for data preparation, a custom Convolutional Neural Network (CNN) model built with TensorFlow, detailed model evaluation, and a Flask web application with a user-friendly UI for making predictions.

## Project Structure

- `prepare_data.py`: Script to process and prepare the raw image dataset.
- `train_model.py`: Script to define, train, and save the CNN model.
- `evaluate_model.py`: Script to evaluate the trained model's performance and generate metrics.
- `app/`: Directory containing the Flask web application.
- `tests/`: Directory containing Pytest tests for the Flask backend.
- `saved_model/`: Directory where the trained model is saved.
- `evaluation_results/`: Directory where evaluation plots and reports are saved.
- `requirements.txt`: A file listing all the necessary Python packages for the project.

---

## How to Run the Project

Here are the steps to get the application running on your own machine.

### Step 1: Set up the Environment

First, install all the necessary libraries using the `requirements.txt` file. It's recommended to do this in a virtual environment.

```bash
pip install -r requirements.txt
```

### Step 2: Download and Place the Dataset

This project uses the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle. You will need to download it manually.

1.  **Go to the dataset page:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2.  **Download** the dataset (you will need a Kaggle account).
3.  **Unzip** the downloaded file (`archive.zip`).
4.  Place the `chest_xray` directory inside the `data/` directory in the project root. The final folder structure should look like this:
    ```
    .
    ├── data/
    │   └── chest_xray/
    │       ├── train/
    │       ├── test/
    │       └── val/
    └── ... (other project files)
    ```

### Step 3: Prepare the Data

Run the data preparation script. This will read all the images, resize and normalize them, and save the processed data as NumPy arrays in the `data/` directory.

```bash
python prepare_data.py
```

### Step 4: Train the Model

Now, run the training script. This will build the CNN, train it on the processed data, and save the best-performing model to the `saved_model/` directory.

```bash
python train_model.py
```
*(Note: This step can be computationally intensive and may take a while depending on your hardware.)*

### Step 5: (Optional) Evaluate the Model

To see the detailed performance metrics (like the confusion matrix and classification report) and generate the plots, run the evaluation script. The results will be saved in the `evaluation_results/` directory.

```bash
python evaluate_model.py
```

### Step 6: Run the Web Application

Finally, start the Flask web server.

```bash
python app/main.py
```

Once the server is running, open your web browser and navigate to `http://127.0.0.1:5000`. You will see the web interface and can start uploading images for prediction.

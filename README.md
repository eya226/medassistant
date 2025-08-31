# AI Brain Tumor Triage System

## Overview

This project is a full-stack deep learning application designed to function as an intelligent triage system for brain tumors. It analyzes brain MRI scans to detect and classify multiple tumor types, then presents the results in a prioritized list to help medical professionals focus on the most urgent cases first.

## Features

*   **Multi-Class Tumor Classification:** Classifies brain MRI scans into 4 categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.
*   **Triage-Based Sorting:** The application accepts multiple scans at once and sorts the results based on clinical urgency (Glioma > Meningioma > Pituitary > No Tumor).
*   **Explainable AI (Grad-CAM):** For tumor diagnoses, the application generates a visual heatmap on the MRI scan to highlight the regions that most influenced the model's decision.
*   **Web Interface:** A clean, user-friendly web application for batch uploading images and viewing the sorted triage list.

## Technology Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** TensorFlow, Keras, OpenCV
*   **Frontend:** HTML, CSS, JavaScript
*   **Testing:** Pytest

---

## Setup and Launch Instructions

### 1. Prerequisites

*   **Python 3.11:** This project is developed and tested with Python 3.11.
*   **Git:** For cloning the repository.

### 2. Set Up a Virtual Environment

From the project's root directory:
```bash
# Create the virtual environment
python -m venv venv
# Activate it (venv\Scripts\activate on Windows, source venv/bin/activate on macOS/Linux)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

This project uses the "Brain Tumor MRI Dataset" from Kaggle.

1.  **Go to the dataset page:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2.  **Download** the dataset.
3.  **Unzip** the file. You will have `Training` and `Testing` folders.
4.  Create a `data/Brain Tumor MRI` directory in your project root. Place the `Training` and `Testing` folders inside it.

### 5. Prepare the Data

This script merges the original training and testing sets and creates a new, robust `train/val/test` split.
```bash
python prepare_data.py
```

### 6. Train the Model

This script trains the model on the prepared data. The best model will be saved in the `saved_model/` directory.
```bash
python train_model.py
```

### 7. Launch the Application

Once the model is trained, launch the web application.
```bash
python app/main.py
```
The server will start. Open your web browser and go to: **http://127.0.0.1:5000**

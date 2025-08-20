import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define constants
DATA_DIR = 'data'
MODEL_OUTPUT_DIR = 'saved_model'
IMG_SIZE = 128
EPOCHS = 20  # Increased epochs, but EarlyStopping will prevent overfitting
BATCH_SIZE = 32

def load_data():
    """Loads the preprocessed data from .npy files."""
    try:
        X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except FileNotFoundError:
        print("Error: Preprocessed data not found.")
        print("Please run 'prepare_data.py' first to generate the data.")
        return None, None, None

def build_model():
    """Builds the CNN model architecture."""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten and Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout for regularization
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Load the data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    if X_train is not None:
        # Build the model
        model = build_model()
        model.summary()

        # Create callbacks
        if not os.path.exists(MODEL_OUTPUT_DIR):
            os.makedirs(MODEL_OUTPUT_DIR)

        # Checkpoint to save the best model
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODEL_OUTPUT_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5, # Stop after 5 epochs of no improvement
            verbose=1,
            restore_best_weights=True
        )

        # Train the model
        print("\nStarting model training...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping]
        )

        # Save the final model
        model.save(os.path.join(MODEL_OUTPUT_DIR, 'final_model.keras'))

        print("\nModel training complete.")
        print(f"Best model saved to '{os.path.join(MODEL_OUTPUT_DIR, 'best_model.keras')}'")
        print(f"Final model saved to '{os.path.join(MODEL_OUTPUT_DIR, 'final_model.keras')}'")

        # Evaluate the model on the test set
        print("\nEvaluating model on the test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}')

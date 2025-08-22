import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
DATA_DIR = 'data'
MODEL_OUTPUT_DIR = 'saved_model'
IMG_SIZE = 128
EPOCHS = 25 # More epochs for fine-tuning, but EarlyStopping will manage it
BATCH_SIZE = 32

def load_data():
    """Loads the preprocessed 3-channel data from .npy files."""
    try:
        X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except FileNotFoundError as e:
        print(f"Error: Preprocessed data not found ({e}).")
        print("Please run 'prepare_data.py' first to generate the data.")
        return None, None, None

def build_transfer_model():
    """Builds the CNN model using DenseNet121 for transfer learning."""
    # Load the base model with pre-trained weights
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Freeze the layers of the base model
    base_model.trainable = False

    # Add a custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Load the data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    if X_train is not None:
        # Build the model
        model = build_transfer_model()
        model.summary()

        # Create an ImageDataGenerator for data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Create the generator for the training data
        train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

        # Create callbacks
        if not os.path.exists(MODEL_OUTPUT_DIR):
            os.makedirs(MODEL_OUTPUT_DIR)

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODEL_OUTPUT_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5, # Stop after 5 epochs of no improvement
            verbose=1,
            restore_best_weights=True
        )

        # Train the model using the generator
        print("\nStarting model training with data augmentation...")
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
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

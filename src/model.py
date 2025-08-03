import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# Set constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "image_classifier_model.h5")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")


def get_data_generators(train_dir, test_dir):
    """Creates ImageDataGenerators for training and testing datasets."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_gen, test_gen


def build_model(num_classes):
    """Builds and returns a CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_save_model(train_dir, test_dir):
    print("🔁 Starting training...")

    train_gen, test_gen = get_data_generators(train_dir, test_dir)
    print("✅ Data generators created")

    model = build_model(num_classes=train_gen.num_classes)
    print("✅ Model built")

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS
    )
    print("✅ Training complete")

    # Ensure model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"📁 Created directory: {MODEL_DIR}")

    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Save class indices
    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(train_gen.class_indices, f)
    print(f"✅ Class indices saved to {CLASS_INDEX_PATH}")

    return model, history


if __name__ == "__main__":
    train_dir = "data/train"  # Make sure these folders exist
    test_dir = "data/test"
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        train_and_save_model(train_dir, test_dir)
    else:
        print("❌ Error: Training or testing directory does not exist.")

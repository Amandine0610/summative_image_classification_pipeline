import tensorflow as tf

old_model_path = 'models/image_classifier_model.h5'

# Load the existing model
model = tf.keras.models.load_model(old_model_path)

# Save in .keras format
new_model_path_keras = 'models/image_classifier_model.keras'
model.save(new_model_path_keras)  # Works fine for Keras format

# Save in SavedModel format using model.export()
new_model_path_tf_folder = 'models/saved_model'
model.export(new_model_path_tf_folder)  # Correct for SavedModel format

print(f"Model re-saved to:\n - {new_model_path_keras}\n - {new_model_path_tf_folder}")

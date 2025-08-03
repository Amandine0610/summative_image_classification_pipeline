# prediction_fixes.py
# Common issues and fixes for image classification predictions

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

class PredictionDebugger:
    """
    A class to help debug and fix common prediction issues
    """
    
    @staticmethod
    def check_preprocessing_consistency(image_path_or_array, target_size=(224, 224)):
        """
        Check if preprocessing is consistent between training and inference
        """
        issues = []
        
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
        
        original_shape = image.shape
        
        # Check 1: Image channels
        if len(original_shape) != 3 or original_shape[2] != 3:
            issues.append(f"❌ Image should have 3 channels (RGB), got shape: {original_shape}")
        
        # Check 2: Resize
        resized = cv2.resize(image, target_size)
        if resized.shape[:2] != target_size:
            issues.append(f"❌ Resize failed. Expected {target_size}, got {resized.shape[:2]}")
        
        # Check 3: Normalization
        normalized = resized.astype(np.float32) / 255.0
        if normalized.min() < 0 or normalized.max() > 1:
            issues.append(f"❌ Normalization issue. Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        # Check 4: Data type
        if normalized.dtype != np.float32:
            issues.append(f"❌ Wrong data type. Expected float32, got {normalized.dtype}")
        
        return issues, {
            'original_shape': original_shape,
            'resized_shape': resized.shape,
            'normalized_range': (normalized.min(), normalized.max()),
            'normalized_mean': normalized.mean(),
            'normalized_std': normalized.std()
        }
    
    @staticmethod
    def compare_preprocessing_methods(image, method1_func, method2_func):
        """
        Compare two different preprocessing methods
        """
        result1 = method1_func(image)
        result2 = method2_func(image)
        
        diff = np.abs(result1 - result2)
        
        return {
            'max_difference': diff.max(),
            'mean_difference': diff.mean(),
            'are_identical': np.allclose(result1, result2, atol=1e-6),
            'result1_stats': {
                'shape': result1.shape,
                'range': (result1.min(), result1.max()),
                'mean': result1.mean()
            },
            'result2_stats': {
                'shape': result2.shape,
                'range': (result2.min(), result2.max()),
                'mean': result2.mean()
            }
        }
    
    @staticmethod
    def check_model_input_compatibility(model, sample_input):
        """
        Check if the input is compatible with the model
        """
        issues = []
        
        # Check input shape
        expected_shape = model.input_shape
        actual_shape = sample_input.shape
        
        if len(actual_shape) != len(expected_shape):
            issues.append(f"❌ Shape dimension mismatch. Expected {len(expected_shape)} dims, got {len(actual_shape)}")
        
        # Check batch dimension
        if actual_shape[0] != 1 and expected_shape[0] is not None:
            issues.append(f"❌ Batch size mismatch. Expected {expected_shape[0]}, got {actual_shape[0]}")
        
        # Check spatial dimensions
        if len(actual_shape) >= 3:
            if actual_shape[1:3] != expected_shape[1:3]:
                issues.append(f"❌ Spatial dimensions mismatch. Expected {expected_shape[1:3]}, got {actual_shape[1:3]}")
        
        # Check channels
        if len(actual_shape) == 4:
            if actual_shape[3] != expected_shape[3]:
                issues.append(f"❌ Channel mismatch. Expected {expected_shape[3]}, got {actual_shape[3]}")
        
        return issues
    
    @staticmethod
    def analyze_prediction_distribution(predictions, class_names):
        """
        Analyze the prediction distribution for anomalies
        """
        analysis = {}
        
        # Check if probabilities sum to 1
        prob_sum = np.sum(predictions)
        analysis['probabilities_sum'] = prob_sum
        analysis['sum_close_to_1'] = np.isclose(prob_sum, 1.0, atol=1e-6)
        
        # Check for NaN or infinite values
        analysis['has_nan'] = np.isnan(predictions).any()
        analysis['has_inf'] = np.isinf(predictions).any()
        
        # Check for negative probabilities
        analysis['has_negative'] = (predictions < 0).any()
        
        # Calculate entropy (measure of uncertainty)
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        max_entropy = np.log(len(predictions))
        analysis['entropy'] = entropy
        analysis['normalized_entropy'] = entropy / max_entropy
        
        # Confidence metrics
        analysis['max_confidence'] = predictions.max()
        analysis['top_2_gap'] = sorted(predictions, reverse=True)[0] - sorted(predictions, reverse=True)[1]
        
        # Class distribution
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        analysis['top_3_predictions'] = [
            (class_names[i], predictions[i]) for i in top_3_indices
        ]
        
        return analysis

# Fixed preprocessing functions

def preprocess_image_fixed_v1(image_path):
    """
    Version 1: Standard OpenCV preprocessing (matches your FastAPI)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def preprocess_image_fixed_v2(image_pil):
    """
    Version 2: PIL-based preprocessing (for Streamlit)
    """
    # Convert PIL to numpy array
    image = np.array(image_pil)
    
    # Ensure RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image_pil.mode != 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def preprocess_image_tensorflow_style(image):
    """
    Version 3: TensorFlow/Keras style preprocessing
    """
    # Resize using TensorFlow
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Resize
    image_resized = tf.image.resize(image_tensor, [224, 224])
    
    # Normalize
    image_normalized = image_resized / 255.0
    
    return image_normalized.numpy()

# Common issues and their fixes

COMMON_ISSUES = {
    "low_confidence": {
        "description": "Model predictions have very low confidence (<30%)",
        "possible_causes": [
            "Model wasn't trained well",
            "Input image doesn't match training data distribution",
            "Preprocessing inconsistency between training and inference",
            "Wrong model architecture for the task"
        ],
        "fixes": [
            "Retrain the model with more data",
            "Check preprocessing pipeline consistency",
            "Verify input image quality and format",
            "Use data augmentation during training"
        ]
    },
    
    "wrong_predictions": {
        "description": "Model makes obviously wrong predictions",
        "possible_causes": [
            "Preprocessing mismatch (training vs inference)",
            "Wrong normalization (e.g., [0,1] vs [-1,1])",
            "Channel order mismatch (RGB vs BGR)",
            "Wrong input size",
            "Model overfitted to training data"
        ],
        "fixes": [
            "Ensure identical preprocessing for training and inference",
            "Check normalization ranges",
            "Verify color channel order",
            "Confirm input image dimensions",
            "Validate model on held-out test set"
        ]
    },
    
    "inconsistent_results": {
        "description": "Same image gives different predictions",
        "possible_causes": [
            "Random data augmentation during inference",
            "Model has dropout layers enabled during inference",
            "Floating point precision issues",
            "Threading issues in prediction"
        ],
        "fixes": [
            "Disable data augmentation during inference",
            "Set model to evaluation mode",
            "Use consistent random seeds",
            "Ensure thread-safe prediction calls"
        ]
    },
    
    "preprocessing_errors": {
        "description": "Images are not processed correctly",
        "possible_causes": [
            "Wrong image loading method",
            "Incorrect resize interpolation",
            "Missing color space conversion",
            "Wrong data type conversion"
        ],
        "fixes": [
            "Use consistent image loading libraries",
            "Specify interpolation method explicitly",
            "Always convert to RGB format",
            "Ensure float32 data type for model input"
        ]
    }
}

def diagnose_prediction_issues(model, image, class_names, expected_class=None):
    """
    Comprehensive diagnosis of prediction issues
    """
    debugger = PredictionDebugger()
    
    # 1. Check preprocessing
    issues, preprocessing_info = debugger.check_preprocessing_consistency(image)
    
    # 2. Preprocess image
    if isinstance(image, str):
        processed = preprocess_image_fixed_v1(image)
    else:
        processed = preprocess_image_fixed_v2(image)
    
    processed_batch = np.expand_dims(processed, axis=0)
    
    # 3. Check model compatibility
    model_issues = debugger.check_model_input_compatibility(model, processed_batch)
    
    # 4. Make prediction
    predictions = model.predict(processed_batch, verbose=0)[0]
    
    # 5. Analyze predictions
    pred_analysis
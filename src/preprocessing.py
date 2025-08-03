import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image with better error handling"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image while maintaining aspect ratio
            image = self.resize_with_padding(image, self.img_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def resize_with_padding(self, image, target_size):
        """Resize image while maintaining aspect ratio and adding padding"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return padded
    
    def load_dataset_from_directory(self, data_dir):
        """Load dataset from directory structure with better validation"""
        images = []
        labels = []
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return np.array([]), np.array([])
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        if not class_dirs:
            logger.warning(f"No class directories found in {data_dir}")
            return np.array([]), np.array([])
        
        logger.info(f"Found classes: {class_dirs}")
        
        # Track class distribution
        class_counts = {}
        
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
            
            logger.info(f"Processing {len(image_files)} images from class '{class_name}'")
            class_counts[class_name] = 0
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                image = self.load_and_preprocess_image(image_path)
                
                if image is not None:
                    images.append(image)
                    labels.append(class_name)
                    class_counts[class_name] += 1
        
        if not images:
            logger.warning("No valid images found")
            return np.array([]), np.array([])
        
        # Log class distribution
        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} images")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Loaded {len(images)} images with {len(set(labels))} classes")
        return images, labels_encoded
    
    def create_more_realistic_sample_dataset(self, output_dir):
        """Create a more realistic sample dataset for demonstration"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sample classes with more realistic patterns
        classes = {
            'cats': {'base_color': (200, 150, 100), 'patterns': ['stripes', 'spots']},
            'dogs': {'base_color': (139, 69, 19), 'patterns': ['solid', 'patches']},
            'birds': {'base_color': (70, 130, 180), 'patterns': ['feathers', 'wings']},
            'airplane': {'base_color': (192, 192, 192), 'patterns': ['wings', 'body']},
            'automobile': {'base_color': (255, 0, 0), 'patterns': ['rectangular', 'wheels']},
            'truck': {'base_color': (0, 100, 0), 'patterns': ['large_body', 'wheels']},
            'horse': {'base_color': (139, 90, 43), 'patterns': ['body', 'legs']}
        }
        
        for class_name, properties in classes.items():
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate sample images for each class
            for i in range(100):  # 100 images per class for better training
                image = self.generate_class_specific_sample(class_name, properties)
                
                # Save image
                image_path = os.path.join(class_dir, f'{class_name}_{i:03d}.jpg')
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, image_bgr)
        
        logger.info(f"Created more realistic sample dataset in {output_dir}")
    
    def generate_class_specific_sample(self, class_name, properties):
        """Generate more realistic class-specific sample images"""
        # Create PIL image for better drawing capabilities
        img = Image.new('RGB', self.img_size, color=(50, 50, 50))  # Dark background
        draw = ImageDraw.Draw(img)
        
        base_color = properties['base_color']
        
        if class_name in ['cats', 'dogs', 'horse']:
            # Animal-like shapes
            self.draw_animal_shape(draw, base_color, class_name)
        elif class_name == 'birds':
            # Bird-like shapes
            self.draw_bird_shape(draw, base_color)
        elif class_name in ['airplane', 'automobile', 'truck']:
            # Vehicle shapes
            self.draw_vehicle_shape(draw, base_color, class_name)
        
        # Add some noise and texture
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert back to numpy array
        return np.array(img)
    
    def draw_animal_shape(self, draw, base_color, animal_type):
        """Draw animal-like shapes"""
        w, h = self.img_size
        
        # Add random variations to color
        color_var = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
        
        if animal_type == 'cats':
            # Cat-like shape with pointed ears
            # Body
            draw.ellipse([w//4, h//3, 3*w//4, 2*h//3], fill=color_var)
            # Head
            draw.ellipse([w//3, h//6, 2*w//3, h//2], fill=color_var)
            # Ears
            draw.polygon([(w//3, h//4), (w//3 + 20, h//6), (w//3 + 40, h//4)], fill=color_var)
            draw.polygon([(2*w//3 - 40, h//4), (2*w//3 - 20, h//6), (2*w//3, h//4)], fill=color_var)
            
        elif animal_type == 'dogs':
            # Dog-like shape with floppy ears
            # Body
            draw.ellipse([w//4, h//3, 3*w//4, 2*h//3], fill=color_var)
            # Head
            draw.ellipse([w//3, h//6, 2*w//3, h//2], fill=color_var)
            # Floppy ears
            draw.ellipse([w//4, h//5, w//3 + 10, h//3], fill=color_var)
            draw.ellipse([2*w//3 - 10, h//5, 3*w//4, h//3], fill=color_var)
            
        elif animal_type == 'horse':
            # Horse-like shape
            # Body
            draw.ellipse([w//5, h//3, 4*w//5, 2*h//3], fill=color_var)
            # Head/neck
            draw.ellipse([w//6, h//8, w//3, h//2], fill=color_var)
            # Legs
            for i in range(4):
                x = w//4 + i * w//8
                draw.rectangle([x, 2*h//3, x + 10, 5*h//6], fill=color_var)
    
    def draw_bird_shape(self, draw, base_color):
        """Draw bird-like shapes"""
        w, h = self.img_size
        color_var = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
        
        # Body
        draw.ellipse([w//3, h//3, 2*w//3, 2*h//3], fill=color_var)
        # Head
        draw.ellipse([w//2 - 30, h//6, w//2 + 30, h//3 + 10], fill=color_var)
        # Wings
        draw.ellipse([w//6, h//2 - 20, w//3 + 20, h//2 + 40], fill=color_var)
        draw.ellipse([2*w//3 - 20, h//2 - 20, 5*w//6, h//2 + 40], fill=color_var)
        # Tail
        draw.polygon([(w//3, 2*h//3), (w//6, 3*h//4), (w//3, 5*h//6)], fill=color_var)
    
    def draw_vehicle_shape(self, draw, base_color, vehicle_type):
        """Draw vehicle-like shapes"""
        w, h = self.img_size
        color_var = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
        
        if vehicle_type == 'airplane':
            # Airplane body
            draw.ellipse([w//6, 2*h//5, 5*w//6, 3*h//5], fill=color_var)
            # Wings
            draw.rectangle([w//8, h//2 - 5, 7*w//8, h//2 + 5], fill=color_var)
            # Tail
            draw.polygon([(5*w//6, 2*h//5), (w - 20, h//3), (5*w//6, 3*h//5)], fill=color_var)
            
        elif vehicle_type == 'automobile':
            # Car body
            draw.rectangle([w//6, 2*h//5, 5*w//6, 3*h//5], fill=color_var)
            # Roof
            draw.rectangle([w//4, h//3, 3*w//4, 2*h//5], fill=color_var)
            # Wheels
            draw.ellipse([w//5, 3*h//5, w//5 + 40, 3*h//5 + 40], fill=(50, 50, 50))
            draw.ellipse([4*w//5 - 40, 3*h//5, 4*w//5, 3*h//5 + 40], fill=(50, 50, 50))
            
        elif vehicle_type == 'truck':
            # Truck body (larger)
            draw.rectangle([w//8, h//3, 7*w//8, 2*h//3], fill=color_var)
            # Cab
            draw.rectangle([w//8, h//4, w//3, 2*h//5], fill=color_var)
            # Wheels
            draw.ellipse([w//6, 2*h//3, w//6 + 50, 2*h//3 + 50], fill=(50, 50, 50))
            draw.ellipse([2*w//3, 2*h//3, 2*w//3 + 50, 2*h//3 + 50], fill=(50, 50, 50))
    
    def augment_data(self, images, labels):
        """Apply comprehensive data augmentation"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            rescale=None  # We already normalized
        )
        
        return datagen.flow(images, labels, batch_size=self.batch_size)
    
    def balance_dataset(self, images, labels):
        """Balance dataset by oversampling minority classes"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)
        
        balanced_images = []
        balanced_labels = []
        
        for label in unique_labels:
            label_mask = labels == label
            label_images = images[label_mask]
            current_count = np.sum(label_mask)
            
            # Add all existing images
            balanced_images.extend(label_images)
            balanced_labels.extend([label] * current_count)
            
            # Oversample to reach max_count
            if current_count < max_count:
                needed = max_count - current_count
                indices = np.random.choice(len(label_images), needed, replace=True)
                balanced_images.extend(label_images[indices])
                balanced_labels.extend([label] * needed)
        
        return np.array(balanced_images), np.array(balanced_labels)
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets with stratification"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Log class distribution in each split
        for split_name, split_labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            unique, counts = np.unique(split_labels, return_counts=True)
            logger.info(f"{split_name} distribution: {dict(zip(unique, counts))}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_label_encoder(self, filepath):
        """Save label encoder"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        label_mapping = {str(i): label for i, label in enumerate(self.label_encoder.classes_)}
        with open(filepath, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        logger.info(f"Label encoder saved to {filepath}")
    
    def load_label_encoder(self, filepath):
        """Load label encoder"""
        with open(filepath, 'r') as f:
            label_mapping = json.load(f)
        
        classes = [label_mapping[str(i)] for i in sorted(map(int, label_mapping.keys()))]
        self.label_encoder.classes_ = np.array(classes)
        logger.info(f"Label encoder loaded from {filepath}")

    def validate_dataset(self, images, labels):
        """Validate dataset for common issues"""
        logger.info("Validating dataset...")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(images)) or np.any(np.isinf(images)):
            logger.warning("Found NaN or infinite values in images")
        
        # Check image value ranges
        min_val, max_val = np.min(images), np.max(images)
        logger.info(f"Image value range: [{min_val:.3f}, {max_val:.3f}]")
        
        if min_val < 0 or max_val > 1:
            logger.warning("Images not properly normalized to [0, 1] range")
        
        # Check class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples = np.min(counts)
        max_samples = np.max(counts)
        
        if max_samples / min_samples > 3:
            logger.warning(f"Imbalanced dataset detected. Max: {max_samples}, Min: {min_samples}")
            logger.info("Consider using balance_dataset() method")
        
        logger.info("Dataset validation complete")

if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()
    
    # Create sample dataset if it doesn't exist
    train_dir = "data/train"
    if not os.path.exists(train_dir) or not any(os.listdir(train_dir)):
        logger.info("Creating sample dataset...")
        preprocessor.create_more_realistic_sample_dataset(train_dir)
    
    # Load and preprocess data
    images, labels = preprocessor.load_dataset_from_directory(train_dir)
    
    if len(images) > 0:
        # Validate dataset
        preprocessor.validate_dataset(images, labels)
        
        # Optional: Balance dataset
        # images, labels = preprocessor.balance_dataset(images, labels)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(images, labels)
        
        # Save label encoder
        preprocessor.save_label_encoder("models/label_encoder.json")
        
        print(f"Preprocessing complete!")
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of classes: {len(preprocessor.label_encoder.classes_)}")
        print(f"Classes: {preprocessor.label_encoder.classes_}")
        
        # Show some statistics
        print(f"\nDataset statistics:")
        print(f"Image value range: [{np.min(images):.3f}, {np.max(images):.3f}]")
        print(f"Image shape: {images[0].shape}")
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        for i, (label, count) in enumerate(zip(preprocessor.label_encoder.classes_, counts)):
            print(f"Class {i} ({label}): {count} samples")
    else:
        logger.error("No images loaded. Please check your data directory.")
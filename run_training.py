#!/usr/bin/env python3
"""
Complete training pipeline script for the ML project.
This script runs the entire pipeline from data creation to model training.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from preprocessing import ImagePreprocessor
from model import ImageClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete training pipeline"""
    
    print("=" * 80)
    print("🚀 STARTING ML PIPELINE TRAINING")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Initialize components
        logger.info("Initializing preprocessor and classifier...")
        preprocessor = ImagePreprocessor(img_size=(224, 224))
        
        # Create sample dataset if needed
        train_dir = "data/train"
        if not os.path.exists(train_dir) or not os.listdir(train_dir):
            logger.info("Creating sample dataset...")
            preprocessor.create_sample_dataset(train_dir)
            logger.info("✅ Sample dataset created successfully!")
        else:
            logger.info("✅ Training data already exists")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        images, labels = preprocessor.load_dataset_from_directory(train_dir)
        
        if len(images) == 0:
            logger.error("❌ No images found in training directory!")
            return False
        
        logger.info(f"✅ Loaded {len(images)} images with {len(set(labels))} classes")
        
        # Split data
        logger.info("Splitting data into train/validation/test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(images, labels)
        
        logger.info(f"✅ Data split complete:")
        logger.info(f"   - Training: {len(X_train)} samples")
        logger.info(f"   - Validation: {len(X_val)} samples")
        logger.info(f"   - Test: {len(X_test)} samples")
        
        # Save label encoder
        os.makedirs("models", exist_ok=True)
        preprocessor.save_label_encoder("models/label_encoder.json")
        logger.info("✅ Label encoder saved")
        
        # Initialize classifier
        num_classes = len(preprocessor.label_encoder.classes_)
        classifier = ImageClassifier(img_size=(224, 224), num_classes=num_classes)
        
        # Create model
        logger.info("Creating CNN model...")
        classifier.create_model()
        logger.info("✅ Model architecture created")
        
        # Train model
        logger.info("Starting model training...")
        logger.info("This may take several minutes depending on your hardware...")
        
        history = classifier.train_model(
            X_train, y_train,
            X_val, y_val,
            epochs=30,  # Adjust as needed
            batch_size=32
        )
        
        logger.info("✅ Model training completed!")
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        evaluation_results = classifier.evaluate_model(
            X_test, y_test,
            class_names=preprocessor.label_encoder.classes_
        )
        
        if evaluation_results:
            accuracy = evaluation_results['accuracy']
            logger.info(f"✅ Model evaluation completed!")
            logger.info(f"   - Test Accuracy: {accuracy:.4f}")
            
            # Print per-class performance
            report = evaluation_results['classification_report']
            logger.info("Per-class performance:")
            for class_name in preprocessor.label_encoder.classes_:
                if class_name in report:
                    metrics = report[class_name]
                    logger.info(f"   - {class_name}: P={metrics['precision']:.3f}, "
                              f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        os.makedirs("static", exist_ok=True)
        
        # Training history plot
        classifier.plot_training_history("static/training_history.png")
        
        # Confusion matrix plot
        if evaluation_results:
            import numpy as np
            cm = np.array(evaluation_results['confusion_matrix'])
            classifier.plot_confusion_matrix(
                cm, 
                preprocessor.label_encoder.classes_,
                "static/confusion_matrix.png"
            )
        
        logger.info("✅ Visualizations generated")
        
        # Calculate training time
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Total training time: {training_time}")
        print(f"📊 Final test accuracy: {accuracy:.4f}")
        print(f"📁 Model saved to: models/image_classifier.h5")
        print(f"📈 Visualizations saved to: static/")
        print("\n🚀 Ready to deploy! Run: python src/prediction.py")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
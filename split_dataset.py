import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
source_dir = 'data/train/'
train_dir = 'data/train/'
test_dir = 'data/test/'

# Clear test_dir to avoid residual files
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir, exist_ok=True)

# Get class names (only directories)
class_names = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
print(f"Classes found: {class_names}")

# Split dataset for each class
for cls in class_names:
    class_dir = os.path.join(source_dir, cls)
    if not os.path.isdir(class_dir):
        continue
    
    # Get all image files
    images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
              if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"No images found in {class_dir}. Skipping...")
        continue
    
    # Split into train and test (80-20 split)
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Create class directories in test
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    
    # Move test images
    for img in test_imgs:
        dest_path = os.path.join(test_dir, cls, os.path.basename(img))
        shutil.move(img, dest_path)
    
    print(f"Class {cls}: {len(train_imgs)} training images, {len(test_imgs)} test images")

# Verify split
print("\nVerification:")
total_train = 0
total_test = 0
for cls in class_names:
    train_count = len([f for f in os.listdir(os.path.join(train_dir, cls)) 
                       if os.path.isfile(os.path.join(train_dir, cls, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(os.path.join(train_dir, cls)) else 0
    test_count = len([f for f in os.listdir(os.path.join(test_dir, cls)) 
                      if os.path.isfile(os.path.join(test_dir, cls, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(os.path.join(test_dir, cls)) else 0
    print(f"Class {cls}: Training images = {train_count}, Test images = {test_count}")
    total_train += train_count
    total_test += test_count

print(f"\nTotal: Training images = {total_train}, Test images = {total_test}")

# Check if test set is populated
if total_test == 0:
    print("Error: Test directory is empty. Please ensure sufficient images are available.")

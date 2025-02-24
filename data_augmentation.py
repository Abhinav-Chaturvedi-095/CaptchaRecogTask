import cv2
import pandas as pd
import numpy as np
import os
import albumentations as A
from tqdm import tqdm  # Progress bar
import random

# Define base paths
TRAIN_IMG_BASE_PATH = "/home/abhinav/TASK/dataset/train-images"
AUGMENTED_IMG_BASE_PATH = "/home/abhinav/TASK/dataset/train-images-augmented/"
VAL_IMG_BASE_PATH = "/home/abhinav/TASK/dataset/validation-images/"
TEST_IMG_BASE_PATH = "/home/abhinav/TASK/dataset/test-images/"
NEW_CSV_PATH = "/home/abhinav/TASK/dataset/captcha_data_augmented.csv"

# Ensure augmented image folder exists
os.makedirs(AUGMENTED_IMG_BASE_PATH, exist_ok=True)

# Load original dataset
csv_path = "/home/abhinav/TASK/dataset/captcha_data.csv"
df = pd.read_csv(csv_path)

# Ensure solutions are in 6-digit format
df['solution'] = df['solution'].astype(str).apply(lambda x: x.zfill(6))

# Split dataset into train, val, and test
test_df = df[df['image_path'].str.contains("test-images")].copy()
val_df = df[df['image_path'].str.contains("validation-images")].copy()
train_df = df[~df['image_path'].str.contains("test-images|validation-images")].copy()

print(f" Original Dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Fix train dataset paths (make absolute)
def get_absolute_train_path(image_path):
    return os.path.join(TRAIN_IMG_BASE_PATH, os.path.basename(image_path))

train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(TRAIN_IMG_BASE_PATH, x))

# Define augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=5, p=0.5),  #  Slight Rotation
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),  #  Blurring
    A.RandomBrightnessContrast(p=0.4),  # Contrast Changes
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # Warping
    A.Perspective(scale=(0.05, 0.1), p=0.4),  # Perspective Distortion
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),  # Grid Distortion
    A.MotionBlur(blur_limit=3, p=0.2),  # Motion Blur
    A.ToGray(p=0.1),  # Convert some images to grayscale
])

# Store new dataset (original + augmented)
augmented_data = train_df.to_dict(orient='records')

# Augment training images
print(" Augmenting training images to create dataset...")
for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    original_path = row["image_path"]
    captcha_solution = row["solution"]

    # Ensure image exists before augmenting
    if not os.path.exists(original_path):
        print(f" Missing Image: {original_path}, skipping...")
        continue

    # Load original image
    image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

    # Ensure image is valid
    if image is None:
        print(f" Warning: Cannot read {original_path}. Skipping...")
        continue

    # Resize & Normalize
    image = cv2.resize(image, (200, 50)).astype('float32') / 255.0

    # Generate a new augmented version
    augmented_image = transform(image=image)["image"]

    # Generate new filename
    new_filename = f"aug_{idx}_{random.randint(1000, 9999)}.png"
    new_path = os.path.join(AUGMENTED_IMG_BASE_PATH, new_filename)

    # Save augmented image
    cv2.imwrite(new_path, (augmented_image * 255).astype(np.uint8))

    # Store augmented image info in dataset
    augmented_data.append({"image_path": new_path, "solution": captcha_solution})

# Convert to DataFrame
augmented_train_df = pd.DataFrame(augmented_data)

# Combine Train, Validation, and Test datasets (Validation & Test remain unchanged)
final_df = pd.concat([augmented_train_df, val_df, test_df], ignore_index=True)

# Save CSV
final_df.to_csv(NEW_CSV_PATH, index=False)

print(f" Augmentation complete! New dataset saved at: {NEW_CSV_PATH}")
print(f" Augmented Train Folder: {AUGMENTED_IMG_BASE_PATH}")
print(f" New CSV File: {NEW_CSV_PATH}")
print(f" Final Dataset Sizes: Train={len(augmented_train_df)}, Val={len(val_df)}, Test={len(test_df)}")

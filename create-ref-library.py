import cv2
import os
import numpy as np
import pickle

print("Starting reference library index creation for 20,000 images...")

# --- Configuration ---
PATH_TO_IMAGES = 'mtg_card_library' # Folder with your 20,000 images
INDEX_FILE = 'flann_index.bin'
NAMES_FILE = 'card_names.pkl'
ORB_FEATURES = 1500 # Increased features for better distinction

# --- Initialization ---
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# This will hold all descriptors from all images
all_descriptors = []
# This will map an index in all_descriptors back to a card name
card_names_map = []

# --- Image Processing ---
try:
    image_files = os.listdir(PATH_TO_IMAGES)
    total_images = len(image_files)
    print(f"Found {total_images} images to process.")
except FileNotFoundError:
    print(f"Error: The directory '{PATH_TO_IMAGES}' was not found.")
    exit()

for i, img_name in enumerate(image_files):
    img_path = os.path.join(PATH_TO_IMAGES, img_name)
    # Read the image in grayscale
    img = cv2.imread(img_path, 0)
    
    if img is None:
        print(f"Warning: Could not read image {img_name}. Skipping.")
        continue

    # Find keypoints and descriptors
    kp, des = orb.detectAndCompute(img, None)
    
    if des is not None:
        all_descriptors.append(des)
        # For each descriptor, we store the card's name
        card_name = os.path.splitext(img_name)[0]
        card_names_map.extend([card_name] * len(des))
        
    print(f"Processed image {i + 1}/{total_images}: {img_name}")

if not all_descriptors:
    print("Error: No descriptors were generated. Is the image path correct?")
    exit()

# --- Index Creation ---
print("\nStacking all descriptors into a single array...")
# Vertically stack all descriptors into a single numpy array for FLANN
descriptors_to_index = np.vstack(all_descriptors)

# FLANN parameters for ORB descriptors
# LSH (Locality-Sensitive Hashing) is recommended for binary descriptors like ORB
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6, # 12
                   key_size=12,     # 20
                   multi_probe_level=1) # 2
search_params = dict(checks=50)

print("Building FLANN index... (This may take a while)")
flann = cv2.FlannBasedMatcher(index_params, search_params)
flann.add([np.float32(descriptors_to_index)]) # FLANN requires float32
flann.train()

print(f"Index built successfully. Saving to file: {INDEX_FILE}")
flann.save(INDEX_FILE)

# Save the card names map
with open(NAMES_FILE, 'wb') as f:
    pickle.dump(card_names_map, f)

print(f"Card names map saved to: {NAMES_FILE}")
print("\n--- Indexing Complete! ---")
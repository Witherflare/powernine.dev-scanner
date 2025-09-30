import requests
import os
import json

# --- Configuration ---

# The JSON file containing the card data.
JSON_FILE = "unique-artwork.json"

# The local folder where images will be saved.
# This should match the folder used by your indexing script.
OUTPUT_FOLDER = "mtg-card-library"

# --- Main Script ---

# 1. Create the output folder if it doesn't exist.
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"✅ Output directory is '{OUTPUT_FOLDER}'.")

# 2. Load the card data from the JSON file.
try:
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        card_data_list = json.load(f)
    print(f"🔎 Found {len(card_data_list)} card entries in '{JSON_FILE}'.")
except FileNotFoundError:
    print(f"❌ Error: The file '{JSON_FILE}' was not found. Please place it in the same directory.")
    exit()
except json.JSONDecodeError:
    print(f"❌ Error: The file '{JSON_FILE}' is not a valid JSON file.")
    exit()

total_cards = len(card_data_list)
print(f"🚀 Starting download of {total_cards} images...")

# 3. Loop through each card entry, extract info, and download.
for i, card in enumerate(card_data_list):
    # Use .get() for safe access to nested keys.
    image_uris = card.get('image_uris', {})
    image_uri = image_uris.get('normal')
    tcgplayer_id = card.get('tcgplayer_id')

    # Validate that we got the necessary data.
    if not image_uri or not tcgplayer_id:
        print(f"⚠️ ({i+1}/{total_cards}) Skipping card, missing 'image_uris.normal' or 'tcgplayer_id'.")
        continue

    # Construct the filename and the full path to save the file.
    filename = f"{tcgplayer_id}.jpg"
    save_path = os.path.join(OUTPUT_FOLDER, filename)

    # Check if the file already exists to avoid re-downloading.
    if os.path.exists(save_path):
        # print(f"({i+1}/{total_cards}) Skipping {filename}, already exists.")
        continue

    # 4. Fetch the image from the URI.
    try:
        response = requests.get(image_uri, stream=True, timeout=10)
        
        # Check if the request was successful.
        if response.status_code == 200:
            # Save the image content to the local file.
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ ({i+1}/{total_cards}) Downloaded {filename}")
        else:
            print(f"❌ ({i+1}/{total_cards}) Failed to download {filename} - Status: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ ({i+1}/{total_cards}) Network error for {filename}: {e}")

print("\n--- Download Complete! ---")
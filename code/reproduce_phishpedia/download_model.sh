#!/bin/bash

# Define variables
ZIP_URL="https://zenodo.org/records/14668190/files/used_models.zip?download=1"  # Replace with actual OneDrive download link
ZIP_FILE="models.zip"
TEMP_DIR="temp_extracted"
TARGET_DIR="models"

# Step 1: Download the ZIP file
echo "Downloading models ZIP file..."
wget --no-check-certificate -O "$ZIP_FILE" "$ZIP_URL"

# Step 2: Unzip the file into a temporary directory
echo "Extracting ZIP file..."
mkdir -p "$TEMP_DIR"
unzip -o "$ZIP_FILE" -d "$TEMP_DIR"

# # Step 3: Move the contents of 'models/11_code/repreduced_phishpedia/trained_models' to 'models/'
echo "Moving trained_models to models/"
mkdir -p "$TARGET_DIR"
mv "$TEMP_DIR/11_code/reproduce_phishpedia/trained_models/"* "$TARGET_DIR/"

# Step 4: Clean up
echo "Cleaning up temporary files..."
rm -r "$ZIP_FILE" "$TEMP_DIR"

echo "Download and extraction complete!"

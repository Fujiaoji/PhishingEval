#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./download.sh <folder_name>"
    exit 1
fi

FOLDER_NAME="$1"
TARGET_DIR="data"

mkdir -p "$TARGET_DIR"

URL="https://zenodo.org/records/14804193/files/$FOLDER_NAME.zip?download=1"
FILE_NAME="$FOLDER_NAME.zip"

echo "Downloading $FILE_NAME to $TARGET_DIR/"
wget -O "$TARGET_DIR/$FILE_NAME" "$URL"

echo "Extracting $FILE_NAME..."
unzip -o "$TARGET_DIR/$FILE_NAME" -d "$TARGET_DIR"

rm "$TARGET_DIR/$FILE_NAME"

echo "Download and extraction complete in $TARGET_DIR/"

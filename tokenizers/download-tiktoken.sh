#!/bin/bash
# Download tiktoken encoding files from OpenAI's tiktoken repository
# These are the official tokenizer files used by GPT models

set -e

RESOURCES_DIR="src/test/resources"
TIKTOKEN_DIR="$RESOURCES_DIR/tiktoken"

echo "Downloading tiktoken files..."

# Create directories if they don't exist
mkdir -p "$TIKTOKEN_DIR"

# Base URL for tiktoken files
BASE_URL="https://openaipublic.blob.core.windows.net/encodings"

# Download each encoding file
files=(
    "r50k_base.tiktoken"
    "p50k_base.tiktoken"
    "cl100k_base.tiktoken"
    "o200k_base.tiktoken"
)

for file in "${files[@]}"; do
    if [ -f "$TIKTOKEN_DIR/$file" ]; then
        echo "✓ $file already exists"
    else
        echo "Downloading $file..."
        curl -L -o "$TIKTOKEN_DIR/$file" "$BASE_URL/$file"
        echo "✓ Downloaded $file"
    fi
done

echo ""
echo "All tiktoken files downloaded successfully!"
echo "Location: $TIKTOKEN_DIR/"

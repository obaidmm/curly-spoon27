#!/bin/bash

# Default values
IMAGE_PATH=""
OUTPUT_PATH=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --scan_path)
            IMAGE_PATH="$2"
            shift
            shift
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$IMAGE_PATH" || -z "$OUTPUT_PATH" ]]; then
    echo "Usage: $0 --scan_path <path> --output_path <path>"
    exit 1
fi

echo "Running Python script with:"
echo "  📂 Image Path: $IMAGE_PATH"
echo "  📂 Output Path: $OUTPUT_PATH"

# Run evaluation.py
python3 /Users/obaidmohiuddin/Desktop/curly-spoon27/backend/evaluation.py "$OUTPUT_PATH" "$IMAGE_PATH"

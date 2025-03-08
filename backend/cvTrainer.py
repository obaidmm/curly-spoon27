import os

# Path to the combined label file
combined_label_file = "parking_dataset/val_labels.txt"

# Directory to save individual label files
output_dir = "parking_dataset/val_labels"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the combined label file
with open(combined_label_file, "r") as f:
    lines = f.readlines()

# Process each line in the combined label file
for line in lines:
    parts = line.strip().split()
    if len(parts) < 6:
        print(f"Skipping malformed line: {line.strip()}")
        continue  # Skip empty or invalid lines

    image_name = parts[0]  # Extract the image name
    class_id = int(parts[1]) - 1  # Convert 1 → 0, 2 → 1

    if class_id not in [0, 1]:  # Ensure class IDs are valid
        print(f"Skipping invalid class ID {class_id + 1} in line: {line.strip()}")
        continue

    x_min, y_min, x_max, y_max = map(float, parts[2:6])  # Extract bounding box coordinates

    # Define actual image dimensions (update these to match your dataset)
    image_width, image_height = 640, 640  # Replace with actual dimensions

    # Convert bounding box to YOLO format (normalized values)
    x_center = (x_min + x_max) / (2 * image_width)
    y_center = (y_min + y_max) / (2 * image_height)
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    # Write the annotation to the corresponding label file
    label_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Label conversion completed successfully!")

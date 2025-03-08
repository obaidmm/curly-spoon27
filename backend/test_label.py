import os

# Define paths
labels_path = "/Users/obaidmohiuddin/Downloads/test_data/test_labels.txt"
output_path = "/Users/obaidmohiuddin/Downloads/test_data/converted_labels.txt"

with open(labels_path, "r") as f_in, open(output_path, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        image_id = parts[0]  # Keep image filename
        class_id = parts[1]
        x_center, y_center, width, height = map(float, parts[2:])

        # Convert YOLO format to (x_min, y_min, x_max, y_max)
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)

        f_out.write(f"{image_id} {class_id} {x_min} {y_min} {x_max} {y_max}\n")

print("✅ Labels converted to correct format!")

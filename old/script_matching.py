import os
import json
import pandas as pd
import cv2
import numpy as np

combined_csv_path = r"E:\VitMix\student_masks\combined.csv"
search_folder = r"E:\VitMix\student_data"
output_mask_folder = r"E:\VitMix\masks"

os.makedirs(output_mask_folder, exist_ok=True)

df = pd.read_csv(combined_csv_path)

required_columns = {'filename', 'region_shape_attributes'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The combined CSV file must contain the columns: {required_columns}")

df = df.dropna(subset=['filename', 'region_shape_attributes'])  # Drop rows with missing required data
df['filename'] = df['filename'].str.strip()

if df.empty:
    print("No valid rows found in the CSV file. Exiting.")
    exit()

all_files_in_search_folder = set()
for root, dirs, files in os.walk(search_folder):
    for file in files:
        all_files_in_search_folder.add(file)

def create_mask(image_path, shapes_json, mask_path):
    """
    Creates a single mask for ONE row (which may contain multiple shapes).
    
    Args:
        image_path (str): Path to the original image.
        shapes_json (str): JSON string containing shape definitions (single or array).
        mask_path (str): Where to save the resulting mask (PNG, JPEG, or BMP).
    
    Returns:
        bool: True if mask creation was successful, False otherwise.
    """
    # Load the image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return False
    height, width, _ = image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    try:
        shape_data = json.loads(shapes_json)
        if not isinstance(shape_data, list):
            # If it's not a list, wrap it in a list so we can iterate uniformly
            shape_data = [shape_data]

        # Draw each shape
        for shape in shape_data:
            shape_name = shape.get('name')
            if shape_name == 'polygon':
                pts = np.array(
                    list(zip(shape['all_points_x'], shape['all_points_y'])),
                    dtype=np.int32
                )
                cv2.fillPoly(mask, [pts], 255)

            elif shape_name == 'ellipse':
                center = (int(shape['cx']), int(shape['cy']))
                axes = (int(shape['rx']), int(shape['ry']))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            elif shape_name == 'circle':
                center = (int(shape['cx']), int(shape['cy']))
                radius = int(shape['r'])
                cv2.circle(mask, center, radius, 255, -1)

            elif shape_name == 'rect':
                x = int(shape['x'])
                y = int(shape['y'])
                w = int(shape['width'])
                h = int(shape['height'])
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)

            elif shape_name == 'polyline':
                pts = np.array(
                    list(zip(shape['all_points_x'], shape['all_points_y'])),
                    dtype=np.int32
                )
                # Treat polyline as a filled polygon by connecting the ends
                cv2.fillPoly(mask, [pts], 255)

            else:
                print(f"Unsupported shape type: {shape_name}")

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing shapes: {shapes_json}, error: {e}")
        return False

    # Save the mask
    cv2.imwrite(mask_path, mask)
    return True


total_rows = len(df)
files_found = 0
masks_created = 0
creation_failed = 0
missing_files = 0

print("\nProcessing each row in combined.csv:")
for idx, row in df.iterrows():
    filename = row['filename']
    shape_attributes = row['region_shape_attributes']

    if filename in all_files_in_search_folder:
        files_found += 1

        image_path = None
        for root, dirs, files in os.walk(search_folder):
            if filename in files:
                image_path = os.path.join(root, filename)
                break


        mask_name = f"student_{idx}_{filename}"
        mask_path = os.path.join(output_mask_folder, mask_name)

        success = create_mask(image_path, shape_attributes, mask_path)
        if success:
            masks_created += 1
            print(f"Row {idx}: Mask created for {filename} -> {mask_path}")
        else:
            creation_failed += 1
            print(f"Row {idx}: Failed to create mask for {filename}")
    else:
        missing_files += 1
        print(f"Row {idx}: MISSING -> {filename}")

# Print summary
print("\nSummary Statistics:")
print(f" - Total rows processed          : {total_rows}")
print(f" - Files found                   : {files_found}")
print(f" - Masks successfully created    : {masks_created}")
print(f" - Mask creation failures        : {creation_failed}")
print(f" - Missing files                 : {missing_files}")

import os
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

search_folder = r"E:\VitMix\student_data"
mask_folder = r"E:\VitMix\masks"

image_files = []
for root, _, files in os.walk(search_folder):
    for file in files:
        image_files.append(os.path.join(root, file))

mask_files = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.startswith("student_")]

pairs = []
for image_path in image_files:
    image_filename = os.path.basename(image_path)
    mask_filename = f"student_{image_filename}"
    mask_path = os.path.join(mask_folder, mask_filename)
    if mask_path in mask_files:
        pairs.append((image_path, mask_path))

for image_path, mask_path in pairs:
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(f"Original Image\n{image_path}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Mask\n{mask_path}")
    plt.axis("off")

    plt.show()

    user_input = input("Press Enter to view the next pair, or type 'q' to quit: ").strip()
    if user_input.lower() == 'q':
        print("Exiting visualization.")
        break

    clear_output(wait=True)

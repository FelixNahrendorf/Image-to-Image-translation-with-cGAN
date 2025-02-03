import os
from PIL import Image
from sklearn.model_selection import train_test_split

def organize_and_preprocess(data_dir, processed_dir, split_ratios=(0.8, 0.1, 0.1)):

    # Directories for the processed data
    train_images_dir = os.path.join(processed_dir, "train", "images")
    train_labels_dir = os.path.join(processed_dir, "train", "labels")
    val_images_dir = os.path.join(processed_dir, "val", "images")
    val_labels_dir = os.path.join(processed_dir, "val", "labels")
    test_images_dir = os.path.join(processed_dir, "test", "images")
    test_labels_dir = os.path.join(processed_dir, "test", "labels")

    # Create processed directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    all_images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('jpg')]

    # Split into train, validation, and test sets
    train_images, temp_images = train_test_split(all_images, test_size=(split_ratios[1] + split_ratios[2]))
    val_images, test_images = train_test_split(temp_images, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]))

    # Process and save each split
    def process_and_save(images_path, images_dir, labels_dir):
        for image_path in images_path:

            with Image.open(image_path) as img:

                img_cropped = center_crop(img)

                img_resized = img_cropped.resize((256, 256))

                image_name = os.path.basename(image_path)
                color_output_path = os.path.join(images_dir, image_name)
                bw_output_path = os.path.join(labels_dir, image_name)

                # Save color image
                img_resized.save(color_output_path, format="JPEG")

                # Convert to black-and-white (grayscale) and save
                img_bw = img_resized.convert("L")
                img_bw.save(bw_output_path, format="PNG")

    # Process each dataset split
    process_and_save(train_images, train_images_dir, train_labels_dir)
    process_and_save(val_images, val_images_dir, val_labels_dir)
    process_and_save(test_images, test_images_dir, test_labels_dir)

    print("Data preprocessing complete.")

def center_crop(img):
    # Get dimensions
    width, height = img.size

    # Determine the shorter and longer side
    if width > height:
        new_width = height
        new_height = height
    else:
        new_width = width
        new_height = width

    # Calculate cropping box (center)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Crop the image
    return img.crop((left, top, right, bottom))

if __name__ == "__main__": 
    data_dir = "data/dataset_2/102flowers"  # Directory containing the raw data
    processed_dir = "data/dataset_2/processed"  # Output directory for processed data
    organize_and_preprocess(data_dir, processed_dir)

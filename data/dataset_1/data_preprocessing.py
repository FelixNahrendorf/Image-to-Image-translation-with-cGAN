import os
from PIL import Image
from sklearn.model_selection import train_test_split

def resize_and_save(input_path, output_path, size=(256, 256)):
    """Resizes an image and saves it to the output path."""
    with Image.open(input_path) as img:
        img = img.resize(size, Image.BICUBIC)
        img.save(output_path)

def organize_and_preprocess(data_dir, processed_dir, split_ratios=(0.8, 0.1, 0.1)):
    # Directories for the raw data
    images_dir = os.path.join(data_dir, "CMP_facade_DB_base", "base")
    labels_dir = os.path.join(data_dir, "CMP_facade_DB_base", "base")

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

    # Gather file pairs (.jpg and .png with matching names)
    file_pairs = []
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            image_path = os.path.join(images_dir, file)
            label_path = os.path.join(labels_dir, file.replace(".jpg", ".png"))
            if os.path.exists(label_path):
                file_pairs.append((image_path, label_path))

    # Split into train, validation, and test sets
    train_pairs, temp_pairs = train_test_split(file_pairs, test_size=(split_ratios[1] + split_ratios[2]))
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]))

    # Process and save each split
    def process_and_save(file_pairs, images_dir, labels_dir):
        for image_path, label_path in file_pairs:
            # Define output paths
            image_filename = os.path.basename(image_path)
            label_filename = os.path.basename(label_path)
            output_image_path = os.path.join(images_dir, image_filename)
            output_label_path = os.path.join(labels_dir, label_filename)

            # Resize and save
            resize_and_save(image_path, output_image_path)
            resize_and_save(label_path, output_label_path)

    # Process each dataset split
    process_and_save(train_pairs, train_images_dir, train_labels_dir)
    process_and_save(val_pairs, val_images_dir, val_labels_dir)
    process_and_save(test_pairs, test_images_dir, test_labels_dir)

    print("Data preprocessing complete.")

if __name__ == "__main__": 
    data_dir = "data"  # Directory containing the raw data
    processed_dir = os.path.join(data_dir, "processed")  # Output directory for processed data
    organize_and_preprocess(data_dir, processed_dir)

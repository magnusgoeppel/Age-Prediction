import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from keras.src.legacy.preprocessing.image import ImageDataGenerator

AUGMENTED_DATA_DIR = 'data/augmented_images'


def augment_images(df_subset, target_count, save_dir, label_col):
    created_images = 0

    # Initialize ImageDataGenerator with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    directories = ['data/part1_prepared', 'data/part2_prepared', 'data/part3_prepared']

    # Calculate how many additional images are needed
    if target_count < len(df_subset):
        print("No additional images needed.")
        return 0

    # number of augmented images per original image
    images_needed = min(1000, target_count - len(df_subset))  # create 1000 images MAX for one age-bin
    augment_per_image = images_needed // len(df_subset) + 1

    for index, single_image in tqdm(df_subset.iterrows(), total=df_subset.shape[0], desc="Augmenting images"):
        if images_needed <= created_images:
            continue

        # the fileName is saved in the column 'Unique-Identifier'
        filename = single_image['Unique-Identifier']

        # Search for the image in the specified directories
        img_path = None
        for directory in directories:
            potential_path = os.path.join(directory, filename)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if not img_path:
            print(f"Image '{filename}' not found in the specified directories.")
            continue

        # Load and preprocess the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image '{img_path}'.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)

        # Generate and save augmented images
        try:
            prefix = f"aug_{single_image[label_col]}_{index}"
            for i, augmented_image in enumerate(datagen.flow(
                    image,
                    batch_size=1,
                    save_to_dir=save_dir,
                    save_prefix=prefix,
                    save_format='jpg'
            )):
                created_images += 1
                if i >= augment_per_image:
                    break
        except Exception as e:
            print(f"Error augmenting image '{img_path}': {e}")

    return created_images


def augment_data(df, label_column):
    augmented_images = 0
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)  # create dir, if not existing

    # target_count = max-age-bin
    target_count = df[label_column].value_counts().max()

    # create data for every age-bin
    for label in df[label_column].unique():
        df_subset = df[df[label_column] == label]
        augmented_images += augment_images(df_subset, target_count, AUGMENTED_DATA_DIR, label_column)

    print(f"Augmented {augmented_images} images.")


def get_augmented_data():
    augmented_data = []
    for img_file in os.listdir(AUGMENTED_DATA_DIR):
        img_path = AUGMENTED_DATA_DIR + '/' + img_file
        label = re.search(r"\(\d+-\d+\)", img_file).group(0)
        augmented_data.append({
            'FilePath': img_path,
            'age_bin': label
        })
    return pd.DataFrame(augmented_data)


def update_dataframe(previous_df):
    df_augmented = get_augmented_data()
    df_combined = pd.concat([previous_df, df_augmented], ignore_index=True)
    return df_combined.sample(frac=1).reset_index(drop=True)

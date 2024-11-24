# ------------- Imports ---------------------------------------------------------------------------
from scripts.data_exploration import *
from scripts.image_helper_functions import *
from scripts.data_augmentation import *
from scripts.model_building import *

# ------------- Data exploration before image-preparation -----------------------------------------
file_endings = print_file_endings()

folder_names = ['data/part1', 'data/part2', 'data/part3']
df = get_dataframe(folder_names)
create_csv(df, "unprepared_images.csv")
show_all_plots(df, "Before Image-preprocessing")
print_statistics(df)

# ------------- Image preparation -----------------------------------------------------------------
crop_all_images_multi_threaded()


# ------------- Data exploration after image preparation ------------------------------------------
folder_names = ['data/part1_prepared', 'data/part2_prepared', 'data/part3_prepared']
df_prepared = get_dataframe(folder_names)
create_csv(df_prepared, "prepared_images.csv")
show_all_plots(df_prepared, "After Image-preprocessing")
print_statistics(df_prepared)

# ------------- Data augmentation -----------------------------------------------------------------
# Model requirements for image
ageList = ['(0-4)', '(5-14)', '(15-24)', '(25-34)', '(35-49)', '(50-69)', '(70-117)']
age_bins = [0, 4, 14, 24, 34, 49, 69, 117]  # Age bin edges

df_prepared['age_bin'] = pd.cut(df_prepared['Age'], bins=age_bins, labels=ageList, right=False)
print("Columns in df_prepared:", df_prepared.columns.tolist())
plot_age_bin_counts(df_prepared, "Age-bins before augmentation", ageList)

# perform data augmentation to balance the dataset
augment_data(df_prepared, 'age_bin')

# update the DataFrame with augmented data
df_prepared = update_dataframe(df_prepared)
plot_age_bin_counts(df_prepared, "Age-bins after augmentation", ageList)

# ------------- Model building --------------------------------------------------------------------
model = create_cnn_model(num_classes=len(ageList))
history, val_true_labels, val_predictions = train_cnn_model(model, df_prepared, epochs=12, batch_size=32)

# ------------- Evaluation ------------------------------------------------------------------------
plot_history(history)
plot_confusion_matrix(val_true_labels, val_predictions, ageList)
plot_normalized_confusion_matrix(val_true_labels, val_predictions, ageList)
plot_class_accuracy_histogram(val_true_labels, val_predictions, ageList)

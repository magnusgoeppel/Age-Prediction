from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras import models, layers
from tqdm import tqdm
import os
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SAVE_MODEL_DIR = 'model'


# ------------ Model for Age detection ------------ #
# We have cropped/resized all images to 100x100x1 (grey-scale)
def create_cnn_model(num_classes, input_shape=(100, 100, 1)):
    model = models.Sequential()
    # Input layer with 32 filters, followed by an AveragePooling2D layer
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.AveragePooling2D((2, 2)))
    # Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(256, kernel_size=3, activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    # A GlobalAveragePooling2D layer before going into Dense layers below
    # GlobalAveragePooling2D layer limits outputs to number of filters in last Conv2D layer above (256)
    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Flatten())    #should not be needed
    model.add(layers.Dense(132, activation='relu'))  # Reduces layers to 132 before final output
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# ------------ Training ------------#
def create_data_generators(df_final, img_height=100, img_width=100, batch_size=32, validation_split=0.2):
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_final,
        x_col="FilePath",
        y_col="age_bin",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training"
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df_final,
        x_col="FilePath",
        y_col="age_bin",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation"
    )

    return train_generator, validation_generator


def train_one_epoch(model, train_generator, total_batches):
    batch_num = 0
    for x_batch, y_batch in tqdm(train_generator, total=total_batches, desc="Processing Batches"):
        batch_num += 1
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if batch_num >= total_batches:
            print(f"Training for Epoch done")
            break


def evaluate_model(model, train_generator, validation_generator, history):
    train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    history['loss'].append(train_loss)
    history['accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)

    print(
        f"loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

    return history


def predict_on_validation(model, validation_generator):
    val_predictions = []
    val_true_labels = []

    for x_batch, y_batch in validation_generator:
        predictions = model(x_batch)
        val_predictions.extend(np.argmax(predictions, axis=1))
        val_true_labels.extend(np.argmax(y_batch, axis=1))
        if len(val_predictions) >= len(validation_generator.filenames):
            break

    return val_true_labels, val_predictions


def train_cnn_model(model, df_final, epochs=5, batch_size=32, validation_split=0.2):
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    train_generator, validation_generator = create_data_generators(df_final,
                                                                   batch_size=batch_size,
                                                                   validation_split=validation_split)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # train for one epoch
        total_batches = len(train_generator) + len(validation_generator)
        train_one_epoch(model, train_generator, total_batches)

        # evaluate model
        history = evaluate_model(model, train_generator, validation_generator, history)

    # generate predictions for the validation dataset
    val_true_labels, val_predictions = predict_on_validation(model, validation_generator)

    model.save(SAVE_MODEL_DIR + "/model.h5")  # saving model
    return history, val_true_labels, val_predictions


def plot_history(history):
    # training & validation accuracy
    plt.figure(figsize=(12, 5))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_labels, predictions, class_names):
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def plot_normalized_confusion_matrix(true_labels, predictions, class_names):
    cm = confusion_matrix(true_labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix (Relative Values)')
    plt.show()


def plot_class_accuracy_histogram(true_labels, predictions, class_names):
    cm = confusion_matrix(true_labels, predictions)
    class_accuracies = np.diagonal(cm) / np.sum(cm, axis=1)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_names, y=class_accuracies, palette="Blues_d")
    plt.xlabel('Class')
    plt.ylabel('Relative Accuracy')
    plt.title('Relative Accuracy per Class (Histogram)')
    plt.ylim(0, 1)  # Accuracy range from 0 to 1
    plt.xticks(rotation=45)
    plt.show()

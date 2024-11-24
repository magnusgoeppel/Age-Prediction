# Age Detection 

## Project Overview

This project is designed to classify individuals into predefined age bins using a Convolutional Neural Network (CNN). The pipeline includes data exploration, preprocessing, augmentation, model training, and evaluation. The dataset contains images with metadata encoding the age, gender, race, and timestamp of the individuals.

## Dataset
The UTKFace Dataset is used in this project. It is a large-scale face dataset containing over 20,000 images of faces with metadata such as age, gender, and ethnicity. The dataset is diverse, spanning a wide range of ages, ethnicities, and facial variations. It is publicly available at UTKFace.

## Folder structure

- **`data/`**: Contains raw, prepared, and augmented image datasets.
- **`scripts/`**: Scripts for data exploration, augmentation, and model building.
- **`model/`**: Saved trained models.
- **`README.md`**: Documentation for the repository.

## Project Structure

### 1. **Data Exploration**
The initial step focuses on understanding the dataset’s characteristics. Visualizations and statistical summaries highlight the distribution of key attributes such as age, gender, and race. This phase helps uncover patterns and potential biases in the dataset, offering insights into how data preprocessing and augmentation should be tailored.

### 2. **Image Preprocessing**
To prepare images for training, preprocessing steps like cropping, resizing, and normalization are applied. These steps ensure consistency in input size, enhance computational efficiency, and improve model convergence. Multi-threaded processing is employed to expedite the task, especially for large datasets.

### 3. **Data Augmentation**
Balancing the dataset is critical to avoid model biases toward overrepresented age groups. Augmentation techniques like rotation, flipping, shifting, and scaling are used to generate synthetic images, enriching underrepresented age bins. This step ensures a more uniform distribution across classes, which is essential for improving classification accuracy.

### 4. **Model Building**
A CNN architecture tailored for age classification is developed. The network incorporates convolutional layers for feature extraction, pooling layers to reduce spatial dimensions, and fully connected layers for age bin classification. The model is trained using prepared and augmented data, with hyperparameters optimized for performance.

### 5. **Evaluation and Analysis**
Model performance is assessed through a variety of metrics and visualizations. Confusion matrices highlight misclassifications across age bins, while accuracy histograms provide insights into the relative performance for each class. Additionally, learning curves illustrate how the model’s accuracy and loss evolve over epochs, offering a comprehensive view of training and validation dynamics.


## Results

- Model accuracy and loss over epochs are plotted for both training and validation datasets.
- Confusion matrix shows the classification performance per age bin.
- Class-wise relative accuracy is visualized as a histogram.
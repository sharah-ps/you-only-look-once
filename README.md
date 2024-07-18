# You Only Look Once

## Overview

This project aims to classify handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). The model is built using Keras and TensorFlow, and the dataset is preprocessed to normalize the pixel values and reshape the images. The project includes steps for data augmentation, model training, evaluation, and visualization of predictions.

## Dataset

The dataset consists of 42,000 images of handwritten digits, each of size 28x28 pixels. The images are grayscale, meaning they have a single channel.

### Data Access

The train and test dataset can be accessed and downloaded from the following Google Drive link: [Google Drive Dataset](https://drive.google.com/drive/folders/1_61paubLpjXutt7jyuvjyjRapAdz8mH9?usp=sharing)

### Data Loading

Load the dataset from the provided link and store it in a suitable directory for further processing.

## Preprocessing

1. **Normalization**: The pixel values of the images are divided by 255.0 to normalize them between 0 and 1.
2. **Reshaping**: The images are reshaped to a 4D tensor with dimensions `(42000, 28, 28, 1)` to fit the input shape expected by the CNN.
3. **Label Encoding**: The labels are one-hot encoded using `to_categorical` from `keras.utils`.

## Splitting the Data

The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection` with a test size of 10%.

## Data Visualization

The first few images from the training set are visualized along with their corresponding labels to get an understanding of the data.

## Standardization

The mean and standard deviation of the training data are calculated, and a standardization function is defined to normalize the data using these values.

## Model Architecture

The CNN model is built using `keras.Sequential` with the following layers:

1. **Convolutional Layers**: 
   - 64 filters, kernel size of (3,3), ReLU activation
   - 128 filters, kernel size of (3,3), ReLU activation
   - 256 filters, kernel size of (3,3), ReLU activation

2. **MaxPooling Layers**: 
   - Pool size of (2,2) after each convolutional block

3. **Batch Normalization**: 
   - Added after each MaxPooling layer

4. **Flatten Layer**: 
   - To convert the 3D output to 1D

5. **Dense Layers**: 
   - 512 units, ReLU activation
   - 10 units, softmax activation (output layer)

## Compilation and Training

The model is compiled using `categorical_crossentropy` loss and `adam` optimizer. It is trained with a batch size of 128 and validation split of 20% for 15 epochs.

## Evaluation

The model is evaluated on the test set, and a confusion matrix is generated to visualize the performance. The classification report is also printed to show precision, recall, and F1-score for each class.

## Visualization of Predictions

The predictions on the test set are visualized by plotting the images along with the true and predicted labels.

## Results

The confusion matrix and classification report provide insights into the model's performance on each class.

## Conclusion

This project demonstrates the process of building and training a CNN for digit recognition. The model achieves good accuracy and provides a robust solution for classifying handwritten digits.

## Code

The complete code for this project is provided in the script file. Make sure to update the paths to the dataset as needed.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- keras
- tensorflow

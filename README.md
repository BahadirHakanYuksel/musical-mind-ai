# Musical Instrument Classification using CNN

This project implements a Convolutional Neural Network (CNN) based system to classify musical instruments from audio files using Mel Spectrograms. The model is trained to identify 3 different types of instruments:

- Guitar
- Drum
- Piano

[Türkçe README için buraya tıklayın (Click here for Turkish README)](README_tr.md)

## Project Structure

- `src/instrument_classifier.py`: Main script for extracting Mel Spectrograms, training the CNN model with data augmentation, evaluating performance, and saving the best model.
- `src/predict_instrument.py`: Script for predicting the class of a new audio file using the trained CNN model.
- `src/activation_comparison.py`: Script for comparing different activation functions within the CNN model architecture.
- `requirements.txt`: Package dependencies.
- `data/metadata/Metadata_Train.csv`: CSV file with training data filenames and their corresponding instrument classes.
- `data/metadata/Metadata_Test.csv`: CSV file with test data filenames and their corresponding instrument classes.
- `data/raw/Train_submission/`: Directory containing training audio files.
- `data/raw/Test_submission/`: Directory containing test audio files.
- `results/`: Directory where trained models, evaluation results, and visualizations are saved.

## Features Extracted

The system primarily uses **Mel Spectrograms** derived from the audio files as input for the CNN model. The `instrument_classifier.py` script also includes code for extracting other features like MFCCs, spectral features, ZCR, RMS, Chroma, tempo, and harmonic/percussive components, although these are not directly used as input for the final CNN model in the current main script.

- **Mel Spectrograms**: Time-frequency representation suitable for CNNs.

Data augmentation techniques (noise addition, pitch shifting, time stretching) are applied during training to improve model robustness.

## Model Implemented

The core of the system is a **Convolutional Neural Network (CNN)** designed to process Mel Spectrogram images. The architecture includes:

- Conv2D layers with Batch Normalization and Dropout for feature extraction.
- MaxPooling layers for downsampling.
- Flatten layer to transition to dense layers.
- Dense layers for classification.
- Softmax activation in the output layer for multi-class probability distribution.

The model uses the Adam optimizer and sparse categorical crossentropy loss. Early stopping is employed to prevent overfitting.

## Activation Functions Comparison

The `src/activation_comparison.py` script trains and evaluates the CNN model using different activation functions to determine their impact on performance. The functions compared are:

- ReLU
- Tanh
- Sigmoid
- ELU
- SELU
- Swish

Results (accuracy and training time) are saved in `cnn_activation_results.csv` and visualized in `activation_comparison.png`.

## Test Results and Visualizations

The training process and evaluation generate several output files saved in the `results/` directory (and the root directory for comparison results):

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

Shows the performance of the best CNN model on the test dataset, detailing correct and incorrect predictions per class.

### CNN Training History

![CNN Training History](results/cnn_training_history.png)

Plots the training and validation accuracy/loss over epochs for the CNN model.

### Activation Function Comparison

![Activation Comparison](activation_comparison.png)

Visual comparison of validation accuracy and training time for different activation functions in the CNN.

## Usage

### Training the CNN Model

```bash
python src/instrument_classifier.py
```

This script will:

- Load training metadata.
- Extract Mel Spectrograms with data augmentation for training data.
- Extract Mel Spectrograms for validation data.
- Train the CNN model.
- Evaluate the model on the test set.
- Save the best model (`results/best_model_cnn.h5`), label encoder (`results/label_encoder.pkl`), confusion matrix (`results/confusion_matrix.png`), and training history (`results/cnn_training_history.png`).

### Predicting with the Trained Model

```bash
python src/predict_instrument.py path/to/your/audiofile.wav [--model results/best_model_cnn.h5] [--encoder results/label_encoder.pkl]
```

Loads the trained model and predicts the instrument class for the specified audio file.

### Comparing Activation Functions

```bash
python src/activation_comparison.py
```

This script will:

- Load training data.
- Extract Mel Spectrograms.
- Train and evaluate CNN models with different activation functions.
- Save comparison results to `cnn_activation_results.csv` and `activation_comparison.png`.

## Installation

1.  Clone the repository (if you haven't already).
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: `tensorflow[and-cuda]` requires appropriate NVIDIA drivers and CUDA toolkit installed if you want GPU acceleration._

## Results Files Generated

- `results/best_model_cnn.h5`: The trained CNN model.
- `results/label_encoder.pkl`: The fitted label encoder for instrument classes.
- `results/confusion_matrix.png`: Visualization of the test set performance.
- `results/cnn_training_history.png`: Plot of training/validation accuracy and loss.
- `activation_comparison.png`: Visualization of activation function comparison results.
- `cnn_activation_results.csv`: Table comparing activation function performance.

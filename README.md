# Musical Instrument Classification

This project implements a machine learning system to classify musical instruments from audio files. The model is trained to identify 4 different types of instruments:

- Guitar
- Drum
- Violin
- Piano

## Project Structure

- `instrument_classifier.py`: Main script for extracting features, training multiple models, and evaluating performance
- `predict_instrument.py`: Script for predicting the class of a new audio file using a trained model
- `activation_comparison.py`: Script for comparing different activation functions in neural network models
- `regression_comparison.py`: Script for comparing different regression models for the classification task
- `requirements.txt`: Package dependencies
- `Metadata_Train.csv`: CSV file with training data filenames and their corresponding instrument classes
- `Metadata_Test.csv`: CSV file with test data filenames and their corresponding instrument classes
- `Train_submission/`: Directory containing training audio files
- `Test_submission/`: Directory containing test audio files

## Features Extracted

The system extracts various audio features from the sound files, including:

- Mel-Frequency Cepstral Coefficients (MFCCs)
- Spectral features (centroid, bandwidth, contrast, rolloff)
- Zero crossing rate
- Root Mean Square Energy
- Chroma features
- Mel spectrograms (for CNN models)
- Tempo and beat-related features
- Harmonic and percussive features

## Models Implemented

The system implements and compares multiple machine learning approaches:

1. **Logistic Regression** with multinomial loss
2. **Support Vector Machine (SVM)** with various kernels
3. **Random Forest** classifier
4. **Multi-Layer Perceptron (MLP)** neural network
5. **Convolutional Neural Network (CNN)** for processing mel spectrograms

Each model is optimized through grid search or appropriate training procedures, and the best-performing model is selected for the final classification task.

## Regression Models Comparison

The `regression_comparison.py` script compares different regression models:

- Logistic Regression (Multinomial)
- Logistic Regression (One-vs-Rest)
- One-vs-Rest LogisticRegression
- One-vs-One LogisticRegression
- SGD Classifier (log loss)
- Ridge Classifier

## Activation Functions Comparison

The `activation_comparison.py` script compares different activation functions in neural networks:

- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent)
- Sigmoid
- ELU (Exponential Linear Unit)
- SELU (Scaled Exponential Linear Unit)
- Swish

This comparison is done for both MLP and CNN models to determine which activation function yields the best accuracy.

## Usage

### Training Models

```
python instrument_classifier.py
```

The script will:

- Extract features from audio files
- Train multiple models
- Evaluate and select the best model
- Test on the held-out test set
- Save visualization of results

### Predicting with a Trained Model

```
python predict_instrument.py path_to_audio_file.wav [--model path_to_model]
```

This will load a trained model and predict the instrument class for the given audio file.

### Comparing Activation Functions

```
python activation_comparison.py
```

This will train neural network models with different activation functions and compare their performance.

### Comparing Regression Models

```
python regression_comparison.py
```

This will train and compare different regression models for the classification task.

## Installation

1. Install dependencies:

```
pip install -r requirements.txt
```

## Results

The system will compare the performance of different models and save:

- The best performing model file
- A confusion matrix visualization
- For CNN models, training history graphs
- For activation function comparison, performance graphs
- For regression model comparison, performance graphs

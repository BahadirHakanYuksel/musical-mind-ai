import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import joblib
import time

# Import feature extractor from main script
from instrument_classifier import AudioFeatureExtractor, prepare_data

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_mlp_model(input_shape, num_classes, activation_function, hidden_layers=[100, 50]):
    """Create a Multi-Layer Perceptron model with specified activation function."""
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    
    # Hidden layers
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation_function))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
    
    # Output layer (always uses softmax for multi-class classification)
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn_model(input_shape, num_classes, activation_function):
    """Create a CNN model with specified activation function."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation=activation_function, input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=activation_function),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=activation_function),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=activation_function),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def compare_activations_mlp(X_train, y_train, X_val, y_val, num_classes):
    """Compare different activation functions for MLP model."""
    # List of activation functions to compare
    activation_functions = ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'swish']
    
    # Dictionary to store results
    results = {
        'activation': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'training_time': []
    }
    
    # Train and evaluate models with different activation functions
    for activation in activation_functions:
        print(f"\nTraining MLP with {activation} activation function...")
        
        # Create model
        model = create_mlp_model(X_train.shape[1], num_classes, activation)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model and measure time
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate on training and validation sets
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Store results
        results['activation'].append(activation)
        results['train_accuracy'].append(train_accuracy)
        results['val_accuracy'].append(val_accuracy)
        results['training_time'].append(training_time)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
    
    return pd.DataFrame(results)

def compare_activations_cnn(X_train, y_train, X_val, y_val, num_classes):
    """Compare different activation functions for CNN model."""
    # Reshape data for CNN input
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    
    # List of activation functions to compare
    activation_functions = ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'swish']
    
    # Dictionary to store results
    results = {
        'activation': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'training_time': []
    }
    
    # Train and evaluate models with different activation functions
    for activation in activation_functions:
        print(f"\nTraining CNN with {activation} activation function...")
        
        # Create model
        model = create_cnn_model(input_shape, num_classes, activation)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model and measure time
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate on training and validation sets
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Store results
        results['activation'].append(activation)
        results['train_accuracy'].append(train_accuracy)
        results['val_accuracy'].append(val_accuracy)
        results['training_time'].append(training_time)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
    
    return pd.DataFrame(results)

def plot_results(mlp_results, cnn_results):
    """Plot comparison of activation functions for both models."""
    plt.figure(figsize=(16, 12))
    
    # MLP Validation Accuracy
    plt.subplot(2, 2, 1)
    plt.bar(mlp_results['activation'], mlp_results['val_accuracy'], color='blue')
    plt.title('MLP: Validation Accuracy by Activation Function')
    plt.xlabel('Activation Function')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1)
    
    # MLP Training Time
    plt.subplot(2, 2, 2)
    plt.bar(mlp_results['activation'], mlp_results['training_time'], color='green')
    plt.title('MLP: Training Time by Activation Function')
    plt.xlabel('Activation Function')
    plt.ylabel('Training Time (seconds)')
    
    # CNN Validation Accuracy
    plt.subplot(2, 2, 3)
    plt.bar(cnn_results['activation'], cnn_results['val_accuracy'], color='orange')
    plt.title('CNN: Validation Accuracy by Activation Function')
    plt.xlabel('Activation Function')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1)
    
    # CNN Training Time
    plt.subplot(2, 2, 4)
    plt.bar(cnn_results['activation'], cnn_results['training_time'], color='red')
    plt.title('CNN: Training Time by Activation Function')
    plt.xlabel('Activation Function')
    plt.ylabel('Training Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png')
    print("Saved activation function comparison plot to activation_comparison.png")

def main():
    # Paths to data
    train_metadata_path = 'Metadata_Train.csv'
    train_dir = 'Train_submission'
    
    # Prepare data
    audio_paths, y_encoded, label_encoder = prepare_data(train_metadata_path, train_dir)
    
    # Feature extraction
    feature_extractor = AudioFeatureExtractor()
    
    # Extract features for MLP model
    print("Extracting features for MLP model...")
    X, y = feature_extractor.extract_features_dataset(audio_paths, y_encoded)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Compare activation functions for MLP
    mlp_results = compare_activations_mlp(
        X_train_scaled, y_train, X_val_scaled, y_val, len(label_encoder.classes_))
    
    # Extract mel spectrograms for CNN
    print("\nExtracting mel spectrograms for CNN...")
    X_spec, y_spec = feature_extractor.extract_mel_spectrograms(audio_paths, y_encoded)
    
    # Split spectrograms into train and validation sets
    X_train_spec, X_val_spec, y_train_spec, y_val_spec = train_test_split(
        X_spec, y_spec, test_size=0.2, random_state=42, stratify=y_spec)
    
    # Compare activation functions for CNN
    cnn_results = compare_activations_cnn(
        X_train_spec, y_train_spec, X_val_spec, y_val_spec, len(label_encoder.classes_))
    
    # Print results tables
    print("\nMLP Results:")
    print(mlp_results)
    print("\nCNN Results:")
    print(cnn_results)
    
    # Plot results
    plot_results(mlp_results, cnn_results)
    
    # Save results to CSV
    mlp_results.to_csv('mlp_activation_results.csv', index=False)
    cnn_results.to_csv('cnn_activation_results.csv', index=False)
    print("Saved results to CSV files")

if __name__ == "__main__":
    main() 
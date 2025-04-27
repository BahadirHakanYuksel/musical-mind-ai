import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')

# Constants for feature extraction
SAMPLE_RATE = 22050  # Standard sample rate
DURATION = 3  # We'll use 3 seconds of each audio file
N_MFCC = 13  # Number of MFCC features to extract
N_MELS = 128  # Number of mel bands to generate
N_FFT = 2048  # Length of FFT window
HOP_LENGTH = 512  # Number of samples between frames

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Class for feature extraction
class AudioFeatureExtractor:
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION, n_mfcc=N_MFCC, 
                 n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.samples = duration * sample_rate
    
    def extract_features(self, file_path):
        """Extract audio features from a file path."""
        try:
            # Load audio file with librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # If audio is too short, pad it
            if len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)), 'constant')
            
            # Extract features (a variety to test which works best)
            features = {}
            
            # MFCCs (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_var = np.var(mfccs, axis=1)
            features['mfccs_mean'] = mfccs_mean
            features['mfccs_var'] = mfccs_var
            
            # Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_var'] = np.var(spectral_centroid)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
            features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
            features['spectral_contrast_var'] = np.var(spectral_contrast, axis=1)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_var'] = np.var(spectral_rolloff)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_var'] = np.var(zcr)
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_var'] = np.var(rms)
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_var'] = np.var(chroma, axis=1)
            
            # Mel Spectrogram (for CNN models)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec)
            
            # Tempo and beat-related features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
            
            # Harmonics and Perceptual Features
            harmonics, percussive = librosa.effects.hpss(audio)
            features['harmonic_mean'] = np.mean(harmonics)
            features['percussive_mean'] = np.mean(percussive)
            features['harmonic_var'] = np.var(harmonics)
            features['percussive_var'] = np.var(percussive)
            
            return features
        
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None
    
    def extract_features_dataset(self, audio_paths, labels=None):
        """Extract features from a list of audio files."""
        X = []
        y = []
        
        for i, path in enumerate(audio_paths):
            features = self.extract_features(path)
            if features is not None:
                # Create feature vector (flattening arrays)
                feature_vector = []
                
                # Add all features except mel_spectrogram (we'll use it separately for CNN)
                for key, value in features.items():
                    if key != 'mel_spectrogram':
                        if isinstance(value, np.ndarray):
                            feature_vector.extend(value.flatten())
                        else:
                            feature_vector.append(value)
                
                X.append(feature_vector)
                
                if labels is not None:
                    y.append(labels[i])
            
            # Print progress
            if (i + 1) % 100 == 0 or i == len(audio_paths) - 1:
                print(f"Processed {i + 1}/{len(audio_paths)} files")
        
        return np.array(X), np.array(y) if labels is not None else None
    
    def extract_mel_spectrograms(self, audio_paths, labels=None):
        """Extract mel spectrograms for CNN models."""
        X = []
        y = []
        
        for i, path in enumerate(audio_paths):
            features = self.extract_features(path)
            if features is not None:
                X.append(features['mel_spectrogram'])
                
                if labels is not None:
                    y.append(labels[i])
            
            # Print progress
            if (i + 1) % 100 == 0 or i == len(audio_paths) - 1:
                print(f"Processed {i + 1}/{len(audio_paths)} files for spectrograms")
        
        return np.array(X), np.array(y) if labels is not None else None

# Data preparation and feature extraction
def prepare_data(train_metadata_path, train_dir):
    """Prepare data for model training."""
    # Read metadata CSV file
    metadata = pd.read_csv(train_metadata_path)
    
    # Create full paths to audio files
    audio_paths = [os.path.join(train_dir, filename) for filename in metadata['FileName']]
    labels = metadata['Class'].values
    
    print(f"Found {len(audio_paths)} audio files with {len(np.unique(labels))} unique classes")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Save label encoder for later use
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    return audio_paths, y_encoded, label_encoder

# Model training functions - Other models are commented out
"""
# Function for Logistic Regression model training
def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("\nTraining Logistic Regression model...")
    
    # Parameter grid for grid search
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]
    }
    
    # Create and train the model with grid search
    model = GridSearchCV(LogisticRegression(multi_class='multinomial', random_state=42), 
                        param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Logistic Regression Best Parameters: {model.best_params_}")
    print(f"Logistic Regression Validation Accuracy: {accuracy:.4f}")
    
    return model, accuracy

# Function for SVM model training
def train_svm(X_train, y_train, X_val, y_val):
    print("\nTraining SVM model...")
    
    # Parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    # Create and train the model with grid search
    model = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"SVM Best Parameters: {model.best_params_}")
    print(f"SVM Validation Accuracy: {accuracy:.4f}")
    
    return model, accuracy

# Function for Random Forest model training
def train_random_forest(X_train, y_train, X_val, y_val):
    print("\nTraining Random Forest model...")
    
    # Parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create and train the model with grid search
    model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Random Forest Best Parameters: {model.best_params_}")
    print(f"Random Forest Validation Accuracy: {accuracy:.4f}")
    
    return model, accuracy

# Function for MLP model training
def train_mlp(X_train, y_train, X_val, y_val):
    print("\nTraining MLP model...")
    
    # Parameter grid for grid search
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    
    # Create and train the model with grid search
    model = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"MLP Best Parameters: {model.best_params_}")
    print(f"MLP Validation Accuracy: {accuracy:.4f}")
    
    return model, accuracy
"""

def train_cnn(X_train, y_train, X_val, y_val, num_classes):
    """Train a CNN model on mel spectrograms."""
    print("\nTraining CNN model...")
    
    # Reshape data for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    
    # Create model - Enhanced architecture with more layers and neurons
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased patience
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Increased epochs
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    # Evaluate on validation set
    _, accuracy = model.evaluate(X_val, y_val)
    
    print(f"CNN Validation Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    
    return model, accuracy

# Main function
def main():
    # Paths to data
    train_metadata_path = 'Metadata_Train.csv'
    test_metadata_path = 'Metadata_Test.csv'
    train_dir = 'Train_submission'
    test_dir = 'Test_submission'
    
    # Prepare data
    audio_paths, y_encoded, label_encoder = prepare_data(train_metadata_path, train_dir)
    
    # Feature extraction
    feature_extractor = AudioFeatureExtractor()
    
    # Initialize models and accuracies dictionaries
    models = {}
    accuracies = {}
    
    """
    # Traditional ML models code - commented out
    # Extract features for traditional ML models
    print("Extracting features for traditional ML models...")
    X, y = feature_extractor.extract_features_dataset(audio_paths, y_encoded)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    
    # Logistic Regression
    models['logistic_regression'], accuracies['logistic_regression'] = train_logistic_regression(
        X_train_scaled, y_train, X_val_scaled, y_val)
    
    # SVM
    models['svm'], accuracies['svm'] = train_svm(
        X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Random Forest
    models['random_forest'], accuracies['random_forest'] = train_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val)
    
    # MLP
    models['mlp'], accuracies['mlp'] = train_mlp(
        X_train_scaled, y_train, X_val_scaled, y_val)
    """
    
    # Extract mel spectrograms for CNN
    print("\nExtracting mel spectrograms for CNN...")
    X_spec, y_spec = feature_extractor.extract_mel_spectrograms(audio_paths, y_encoded)
    
    # Split spectrograms into train and validation sets
    X_train_spec, X_val_spec, y_train_spec, y_val_spec = train_test_split(
        X_spec, y_spec, test_size=0.2, random_state=42, stratify=y_spec)
    
    # Train CNN model
    cnn_model, cnn_accuracy = train_cnn(
        X_train_spec, y_train_spec, X_val_spec, y_val_spec, len(label_encoder.classes_))
    
    # Add CNN model to the dictionaries
    models['cnn'] = cnn_model
    accuracies['cnn'] = cnn_accuracy
    
    # We only have CNN model, so it's the best model
    best_model_name = 'cnn'
    best_model = models['cnn']
    best_accuracy = accuracies['cnn']
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save the CNN model
    best_model.save(f'best_model_{best_model_name}.h5')
    print(f"Saved best model as best_model_{best_model_name}.h5")
    
    # Test the model on the test set
    test_metadata = pd.read_csv(test_metadata_path)
    test_paths = [os.path.join(test_dir, filename) for filename in test_metadata['FileName']]
    test_labels = label_encoder.transform(test_metadata['Class'])
    
    # Extract mel spectrograms for test data
    X_test_spec, _ = feature_extractor.extract_mel_spectrograms(test_paths)
    X_test_spec = X_test_spec.reshape(X_test_spec.shape[0], X_test_spec.shape[1], X_test_spec.shape[2], 1)
    test_pred = np.argmax(best_model.predict(X_test_spec), axis=1)
    
    # Print test results
    test_accuracy = accuracy_score(test_labels, test_pred)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_pred, target_names=label_encoder.classes_))
    
    # Plot confusion matrix
    cm = confusion_matrix(test_labels, test_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    print("Saved confusion matrix visualization as confusion_matrix.png")

if __name__ == "__main__":
    main() 
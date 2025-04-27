import os
import sys
import numpy as np
import joblib
import tensorflow as tf
import librosa
import argparse

# Import feature extractor from the main script
from instrument_classifier import AudioFeatureExtractor

def load_model(model_path):
    """Load a saved model."""
    if model_path.endswith('.h5'):
        # CNN model
        model = tf.keras.models.load_model(model_path)
        is_cnn = True
    else:
        # Traditional ML model
        model = joblib.load(model_path)
        is_cnn = False
    
    return model, is_cnn

def predict_instrument(audio_path, model_path=None):
    """Predict the instrument class for a given audio file."""
    # Find the best model if not specified
    if model_path is None:
        # Look for model files in the current directory
        model_files = [f for f in os.listdir() if f.startswith('best_model_')]
        if not model_files:
            print("No saved model found. Please train the model first.")
            return
        
        # Use the first model file found
        model_path = model_files[0]
    
    print(f"Using model: {model_path}")
    
    # Load the model
    model, is_cnn = load_model(model_path)
    
    # Load the label encoder
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Create feature extractor
    feature_extractor = AudioFeatureExtractor()
    
    # Extract features
    features = feature_extractor.extract_features(audio_path)
    
    if features is None:
        print(f"Failed to extract features from {audio_path}")
        return
    
    if is_cnn:
        # For CNN model, use the mel spectrogram
        X = features['mel_spectrogram']
        X = X.reshape(1, X.shape[0], X.shape[1], 1)
        
        # Make prediction
        pred_proba = model.predict(X)[0]
        pred_class = np.argmax(pred_proba)
        
        # Get class probabilities
        class_probabilities = dict(zip(label_encoder.classes_, pred_proba))
    else:
        # For traditional ML models, use the feature vector
        feature_vector = []
        
        # Add all features except mel_spectrogram
        for key, value in features.items():
            if key != 'mel_spectrogram':
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value.flatten())
                else:
                    feature_vector.append(value)
        
        X = np.array([feature_vector])
        
        # Load the scaler and scale the features
        scaler = joblib.load('scaler.pkl')
        X_scaled = scaler.transform(X)
        
        # Make prediction
        pred_class = model.predict(X_scaled)[0]
        
        # Get class probabilities if the model supports it
        try:
            pred_proba = model.predict_proba(X_scaled)[0]
            class_probabilities = dict(zip(label_encoder.classes_, pred_proba))
        except:
            class_probabilities = None
    
    # Get the predicted class name
    predicted_instrument = label_encoder.inverse_transform([pred_class])[0]
    
    print(f"\nPredicted Instrument: {predicted_instrument}")
    
    # Print probabilities if available
    if class_probabilities:
        print("\nClass Probabilities:")
        for cls, prob in sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"{cls}: {prob:.4f}")
    
    return predicted_instrument

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict instrument type from audio file')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    parser.add_argument('--model', type=str, help='Path to the saved model (optional)', default=None)
    
    args = parser.parse_args()
    
    predict_instrument(args.audio_path, args.model) 
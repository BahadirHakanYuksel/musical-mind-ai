import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import torch
from instrument_classifier import AudioFeatureExtractor
from transformer_classifier import AudioTransformerModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf
import librosa

# Sabitler
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def load_data(test_metadata_path, test_dir):
    """Test verilerini yükler"""
    test_metadata = pd.read_csv(test_metadata_path)
    test_audio_paths = [os.path.join(test_dir, filename) for filename in test_metadata['FileName']]
    test_labels = test_metadata['Class'].values
    
    return test_audio_paths, test_labels

def load_cnn_model(model_path, label_encoder_path):
    """CNN modelini yükler"""
    model = load_model(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, label_encoder

def load_transformer_model(model_dir, label_encoder_path):
    """Transformer modelini yükler"""
    label_encoder = joblib.load(label_encoder_path)
    num_labels = len(label_encoder.classes_)
    transformer_model = AudioTransformerModel(num_labels=num_labels)
    transformer_model.load_model(model_dir)
    return transformer_model, label_encoder

def evaluate_cnn_model(model, test_audio_paths, test_labels, label_encoder):
    """CNN modelini değerlendirir"""
    print("CNN modeli değerlendiriliyor...")
    feature_extractor = AudioFeatureExtractor()
    
    # Etiketleri encode et
    y_test_encoded = label_encoder.transform(test_labels)
    
    # Mel spektrogramlarını çıkar
    X_test_spec, _ = feature_extractor.extract_mel_spectrograms(test_audio_paths, augment=False)
    X_test_spec = X_test_spec[..., np.newaxis]  # Kanal boyutunu ekle
    
    # Tahmin
    y_pred_probs = model.predict(X_test_spec)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Değerlendirme
    accuracy = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
    
    print(f"CNN Model Accuracy: {accuracy:.4f}")
    print("CNN Classification Report:")
    print(report)
    
    return y_pred, y_test_encoded, accuracy

def evaluate_transformer_model(transformer_model, test_audio_paths, test_labels, label_encoder):
    """Transformer modelini değerlendirir"""
    print("Transformer modeli değerlendiriliyor...")
    
    # Etiketleri encode et
    y_test_encoded = label_encoder.transform(test_labels)
    
    # Test et - doğrudan audio_paths ve labels kullan
    predictions, ground_truth = transformer_model.evaluate(test_audio_paths, y_test_encoded, batch_size=8)
    
    # Değerlendirme
    accuracy = accuracy_score(ground_truth, predictions)
    report = classification_report(ground_truth, predictions, target_names=label_encoder.classes_)
    
    print(f"Transformer Model Accuracy: {accuracy:.4f}")
    print("Transformer Classification Report:")
    print(report)
    
    return predictions, ground_truth, accuracy

def plot_confusion_matrices(cnn_preds, transformer_preds, true_labels, label_encoder, save_dir):
    """CNN ve Transformer modelleri için karışıklık matrislerini çizer"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # CNN Karışıklık Matrisi
    cm_cnn = confusion_matrix(true_labels, cnn_preds)
    axes[0].imshow(cm_cnn, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('CNN Confusion Matrix')
    
    tick_marks = np.arange(len(label_encoder.classes_))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(label_encoder.classes_, rotation=45)
    axes[0].set_yticklabels(label_encoder.classes_)
    axes[0].set_ylabel('True label')
    axes[0].set_xlabel('Predicted label')
    
    # Transformer Karışıklık Matrisi
    cm_transformer = confusion_matrix(true_labels, transformer_preds)
    axes[1].imshow(cm_transformer, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Transformer Confusion Matrix')
    
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(label_encoder.classes_, rotation=45)
    axes[1].set_yticklabels(label_encoder.classes_)
    axes[1].set_ylabel('True label')
    axes[1].set_xlabel('Predicted label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_confusion_matrix.png'))

def plot_accuracy_comparison(cnn_accuracy, transformer_accuracy, save_dir):
    """Model doğruluk karşılaştırma grafiğini çizer"""
    plt.figure(figsize=(10, 6))
    
    models = ['CNN', 'Transformer']
    accuracies = [cnn_accuracy, transformer_accuracy]
    
    bar_colors = ['tab:blue', 'tab:orange']
    
    plt.bar(models, accuracies, color=bar_colors)
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    
    # Çubukların üzerinde değerleri göster
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_accuracy_comparison.png'))

def main():
    # Veri yolları
    base_dir = os.path.dirname(os.path.dirname(__file__))
    test_metadata_path = os.path.join(base_dir, 'data', 'metadata', 'Metadata_Test.csv')
    test_dir = os.path.join(base_dir, 'data', 'raw', 'Test_submission')
    results_dir = os.path.join(base_dir, 'results')
    cnn_model_path = os.path.join(results_dir, 'best_model_cnn.h5')
    cnn_label_encoder_path = os.path.join(results_dir, 'label_encoder.pkl')
    transformer_model_dir = os.path.join(results_dir, 'transformer', 'model')
    transformer_label_encoder_path = os.path.join(results_dir, 'transformer', 'label_encoder.pkl')
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Veriyi yükle
    test_audio_paths, test_labels = load_data(test_metadata_path, test_dir)
    
    # CNN modelini yükle ve değerlendir
    cnn_model, cnn_label_encoder = load_cnn_model(cnn_model_path, cnn_label_encoder_path)
    cnn_preds, true_labels_cnn, cnn_accuracy = evaluate_cnn_model(
        cnn_model, test_audio_paths, test_labels, cnn_label_encoder
    )
    
    # Transformer modelini yükle ve değerlendir
    transformer_model, transformer_label_encoder = load_transformer_model(
        transformer_model_dir, transformer_label_encoder_path
    )
    transformer_preds, true_labels_transformer, transformer_accuracy = evaluate_transformer_model(
        transformer_model, test_audio_paths, test_labels, transformer_label_encoder
    )
    
    # Karşılaştırma grafikleri oluştur
    plot_confusion_matrices(
        cnn_preds, transformer_preds, true_labels_cnn, cnn_label_encoder, comparison_dir
    )
    
    plot_accuracy_comparison(cnn_accuracy, transformer_accuracy, comparison_dir)
    
    print(f"\n--- Model Karşılaştırma Sonuçları ---")
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    print(f"Transformer Accuracy: {transformer_accuracy:.4f}")
    print(f"Doğruluk Farkı: {transformer_accuracy - cnn_accuracy:.4f}")
    print(f"Yüzde İyileştirme: {((transformer_accuracy / cnn_accuracy) - 1) * 100:.2f}%")
    print(f"\nKarşılaştırma grafikleri {comparison_dir} dizinine kaydedildi.")

if __name__ == "__main__":
    # TensorFlow loglarını bastır
    tf.get_logger().setLevel('ERROR')
    # PyTorch log seviyesini ayarla
    torch.set_printoptions(precision=4)
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import joblib
import time

# Ana betikten özellik çıkarıcıyı içe aktar
from instrument_classifier import AudioFeatureExtractor, prepare_data

# Tekrarlanabilirlik için rastgele tohumları ayarla
np.random.seed(42)
tf.random.set_seed(42)

def create_cnn_model(input_shape, num_classes, activation_function):
    """Belirtilen aktivasyon fonksiyonu ile bir CNN modeli oluşturur."""
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
    
    # Modeli derle
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def compare_activations_cnn(X_train, y_train, X_val, y_val, num_classes):
    """CNN modeli için farklı aktivasyon fonksiyonlarını karşılaştırır."""
    # CNN girişi için veriyi yeniden şekillendir
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    
    # Karşılaştırılacak aktivasyon fonksiyonları listesi
    activation_functions = ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'swish']
    
    # Sonuçları saklamak için sözlük
    results = {
        'activation': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'training_time': []
    }
    
    # Farklı aktivasyon fonksiyonlarıyla modelleri eğit ve değerlendir
    for activation in activation_functions:
        print(f"\n{activation} aktivasyon fonksiyonu ile CNN eğitiliyor...")
        
        # Model oluştur
        model = create_cnn_model(input_shape, num_classes, activation)
        
        # Erken durdurma
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Modeli eğit ve zamanı ölç
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
        
        # Eğitim ve doğrulama setlerinde değerlendir
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Sonuçları sakla
        results['activation'].append(activation)
        results['train_accuracy'].append(train_accuracy)
        results['val_accuracy'].append(val_accuracy)
        results['training_time'].append(training_time)
        
        print(f"Eğitim Doğruluğu: {train_accuracy:.4f}")
        print(f"Doğrulama Doğruluğu: {val_accuracy:.4f}")
        print(f"Eğitim Süresi: {training_time:.2f} saniye")
    
    return pd.DataFrame(results)

def plot_results(cnn_results):
    """CNN modeli için aktivasyon fonksiyonlarının karşılaştırmasını çizer."""
    plt.figure(figsize=(16, 6))
    
    # CNN Doğrulama Doğruluğu
    plt.subplot(1, 2, 1)
    plt.bar(cnn_results['activation'], cnn_results['val_accuracy'], color='orange')
    plt.title('CNN: Aktivasyon Fonksiyonuna Göre Doğrulama Doğruluğu')
    plt.xlabel('Aktivasyon Fonksiyonu')
    plt.ylabel('Doğrulama Doğruluğu')
    plt.ylim(0, 1)
    
    # CNN Eğitim Süresi
    plt.subplot(1, 2, 2)
    plt.bar(cnn_results['activation'], cnn_results['training_time'], color='red')
    plt.title('CNN: Aktivasyon Fonksiyonuna Göre Eğitim Süresi')
    plt.xlabel('Aktivasyon Fonksiyonu')
    plt.ylabel('Eğitim Süresi (saniye)')
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png')
    print("Aktivasyon fonksiyonu karşılaştırma grafiği activation_comparison.png olarak kaydedildi")

def main():
    # Veri yolları
    # Use paths relative to the project root
    train_metadata_path = 'data/metadata/Metadata_Train.csv'
    train_dir = 'data/raw/Train_submission/Train_submission'
    
    # Veriyi hazırla
    audio_paths, y_encoded, label_encoder = prepare_data(train_metadata_path, train_dir)
    
    # Özellik çıkarma
    feature_extractor = AudioFeatureExtractor()
    
    # CNN için mel spektrogramlarını çıkar
    print("\nCNN için mel spektrogramları çıkarılıyor...")
    X_spec, y_spec = feature_extractor.extract_mel_spectrograms(audio_paths, y_encoded)
    
    # Spektrogramları eğitim ve doğrulama setlerine ayır
    X_train_spec, X_val_spec, y_train_spec, y_val_spec = train_test_split(
        X_spec, y_spec, test_size=0.2, random_state=42, stratify=y_spec)
    
    # CNN için aktivasyon fonksiyonlarını karşılaştır
    cnn_results = compare_activations_cnn(
        X_train_spec, y_train_spec, X_val_spec, y_val_spec, len(label_encoder.classes_))
    
    # Sonuç tablolarını yazdır
    print("\nCNN Sonuçları:")
    print(cnn_results)
    
    # Sonuçları çizdir
    plot_results(cnn_results)
    
    # Sonuçları CSV'ye kaydet
    cnn_results.to_csv('cnn_activation_results.csv', index=False)
    print("Sonuçlar CSV dosyalarına kaydedildi")

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import soundfile as sf  # Soundfile kütüphanesini ekle - ses dosyalarını okumak için
from sklearn.preprocessing import LabelEncoder # LabelEncoder'ı tut
from sklearn.model_selection import train_test_split # train_test_split'i tut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Metrikleri tut
import tensorflow as tf # tensorflow içe aktarmasını ekle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization # Spektrogramlar için Conv2D kullan
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')

# Özellik çıkarma için sabitler
SAMPLE_RATE = 22050  # Standart örnekleme oranı
DURATION = 5  # Her ses dosyasının 5 saniyesini kullanacağız
N_MFCC = 13  # Çıkarılacak MFCC özelliklerinin sayısı
N_MELS = 128  # Üretilecek mel bantlarının sayısı
N_FFT = 2048  # FFT penceresinin uzunluğu
HOP_LENGTH = 512  # Kareler arasındaki örnek sayısı

# Tekrarlanabilirlik için rastgele tohumları ayarla
np.random.seed(42)
tf.random.set_seed(42)

# Özellik çıkarma sınıfı
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
        """Bir dosya yolundan ses özelliklerini çıkarır."""
        try:
            # Ses dosyasını librosa ile yükle
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Ses çok kısaysa, doldur
            if len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)), 'constant') # Düzeltilmiş doldurma satırı
            
            # Özellikleri çıkar (hangisinin en iyi çalıştığını test etmek için çeşitli)
            features = {}
            
            # MFCC'ler (Mel-Frekans Kepstral Katsayıları) - Potansiyel gelecek kullanım veya analiz için tut, ancak CNN girişi için değil
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_var = np.var(mfccs, axis=1)
            features['mfccs_mean'] = mfccs_mean
            features['mfccs_var'] = mfccs_var
            
            # Spektral Özellikler - Potansiyel gelecek kullanım veya analiz için tut
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
            
            # Sıfır Geçiş Oranı - Potansiyel gelecek kullanım veya analiz için tut
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_var'] = np.var(zcr)
            
            # Kök Ortalama Kare Enerjisi - Potansiyel gelecek kullanım veya analiz için tut
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_var'] = np.var(rms)
            
            # Kroma Özellikleri - Potansiyel gelecek kullanım veya analiz için tut
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_var'] = np.var(chroma, axis=1)
            
            # Mel Spektrogramı (CNN modelleri için) - Bu CNN için birincil özelliktir
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec) # dB ölçeğine dönüştür
            
            # Tempo ve vuruşla ilgili özellikler - Potansiyel gelecek kullanım veya analiz için tut
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
            
            # Harmonikler ve Algısal Özellikler - Potansiyel gelecek kullanım veya analiz için tut
            harmonics, percussive = librosa.effects.hpss(audio)
            features['harmonic_mean'] = np.mean(harmonics)
            features['percussive_mean'] = np.mean(percussive)
            features['harmonic_var'] = np.var(harmonics)
            features['percussive_var'] = np.var(percussive)
            
            return features
        
        except Exception as e:
            print(f"{file_path} dosyasından özellik çıkarılırken hata oluştu: {e}")
            return None
    
    # Geleneksel ML özellikleri için olduğu için extract_features_dataset'i kaldır

    def extract_mel_spectrograms(self, audio_paths, labels=None, augment=True):
        """CNN modelleri için mel spektrogramlarını çıkarır. İsteğe bağlı olarak gürültü ekler, perde kaydırır ve zaman uzatır."""
        X = []
        y = []
        
        for i, path in enumerate(audio_paths):
            try:
                # İlk olarak soundfile ile yüklemeyi dene
                try:
                    audio, sr = sf.read(path)
                except Exception as e:
                    print(f"Soundfile ile yüklenemedi, librosa deneniyor: {path}")
                    # Soundfile başarısız olursa librosa ile dene
                    audio, sr = librosa.load(path, sr=self.sample_rate, duration=self.duration)
                
                # Stereo dosyaları mono'ya dönüştür (gerekirse)
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                # Örnekleme oranını yeniden örnekle (gerekirse)
                if sr != self.sample_rate:
                    # resample ile oranı ayarla
                    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
                
                # Kırpılmış sesi istenen süreye getir (kısaysa doldur, uzunsa kırp)
                if len(audio) > self.samples:
                    audio = audio[:self.samples]  # Baştan itibaren DURATION kadar al
                elif len(audio) < self.samples:
                    audio = np.pad(audio, (0, self.samples - len(audio)), 'constant')

                # Veri Artırma: augment=True ise rastgele uygula
                if augment:
                    # Gürültü Ekleme (%50 olasılıkla)
                    if np.random.rand() < 0.5:
                        noise_amp = 0.005 * np.random.uniform() * np.amax(audio)  # Gürültü genliğini ayarla
                        audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
                    
                    # Perde Kaydırma (%50 olasılıkla, -2 ile +2 yarım ton arası)
                    if np.random.rand() < 0.5:
                        n_steps = np.random.randint(-2, 3)  # -2, -1, 0, 1, 2
                        if n_steps != 0:  # 0 yarım ton kaydırma anlamsız
                            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
                    
                    # Zaman Uzatma (%50 olasılıkla, 0.9x ile 1.1x arası hız)
                    if np.random.rand() < 0.5:
                        rate = np.random.uniform(0.9, 1.1)
                        if rate != 1.0:  # 1.0 hızında uzatma anlamsız
                            audio = librosa.effects.time_stretch(y=audio, rate=rate)
                            # Zaman uzatma sesin uzunluğunu değiştirir, tekrar kırp/doldur
                            if len(audio) > self.samples:
                                audio = audio[:self.samples]
                            elif len(audio) < self.samples:
                                audio = np.pad(audio, (0, self.samples - len(audio)), 'constant')
                
                # Mel spektrogramını hesapla
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=sr, 
                    n_mels=self.n_mels, 
                    n_fft=self.n_fft, 
                    hop_length=self.hop_length
                )
                mel_spec_db = librosa.power_to_db(mel_spec)  # dB ölçeğine dönüştür
                
                # Mel spektrogramının beklenen şekle sahip olduğundan emin ol
                expected_time_steps = int(np.ceil(self.samples / self.hop_length))
                if mel_spec_db.shape[1] < expected_time_steps:
                    mel_spec_db = np.pad(
                        mel_spec_db, 
                        ((0, 0), (0, expected_time_steps - mel_spec_db.shape[1])), 
                        mode='constant', 
                        constant_values=-80  # Sessizlikle doldur (-80 dB)
                    )
                elif mel_spec_db.shape[1] > expected_time_steps:
                    mel_spec_db = mel_spec_db[:, :expected_time_steps]

                X.append(mel_spec_db)
                
                if labels is not None and i < len(labels):
                    y.append(labels[i])

            except Exception as e:
                print(f"{path} dosyasından spektrogram çıkarılırken hata oluştu: {e}")
                continue  # Hata oluşursa bu dosyayı atla

            # İlerlemeyi yazdır
            if (i + 1) % 100 == 0 or i == len(audio_paths) - 1:
                print(f"Spektrogramlar için {i + 1}/{len(audio_paths)} dosya işlendi (Augment: {augment})")
        
        if not X:  # Eğer boş liste ise
            print("Hiçbir spektrogram çıkarılamadı!")
            return np.array([]), np.array([]) if labels is not None else np.array([])
            
        # Dizilerin boyutlarını yazdır
        X_array = np.array(X)
        y_array = np.array(y) if labels is not None else None
        print(f"X şekli: {X_array.shape}, y şekli: {y_array.shape if y_array is not None else None}")
        
        return X_array, y_array

# Veri hazırlama ve özellik çıkarma
def prepare_data(train_metadata_path, train_dir):
    """Model eğitimi için veriyi hazırlar."""
    # Meta veri CSV dosyasını oku
    metadata = pd.read_csv(train_metadata_path)
    
    # Ses dosyalarına tam yolları oluştur
    audio_paths = [os.path.join(train_dir, filename) for filename in metadata['FileName']]
    labels = metadata['Class'].values
    
    print(f"{len(np.unique(labels))} benzersiz sınıfa sahip {len(audio_paths)} ses dosyası bulundu")
    
    # Etiketleri kodla
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Etiket kodlayıcıyı daha sonra kullanmak üzere kaydet
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    return audio_paths, y_encoded, label_encoder


def train_cnn(X_train, y_train, X_val, y_val, num_classes, results_dir): # results_dir argümanını ekle
    """Mel spektrogramları üzerinde bir CNN modeli eğitir."""
    print("\nCNN modeli eğitiliyor...")
    
    # Girdi şeklini kontrol et ve debug için yazdır
    print(f"Model giriş boyutu: {X_train.shape[1:]}")
    
    # CNN girişi için veriyi yeniden şekillendir (channels_last formatı)
    if len(X_train.shape) == 2:
        # Eğer (örnekler, özellikler) şeklindeyse, yeniden yapılandır
        print("2D girdi tespit edildi, reshape ediliyor...")
        n_samples = X_train.shape[0]
        # Bir spektrogram matrisi (n_mels, time_steps) oluştur
        n_mels = N_MELS
        time_steps = int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH))
        
        # Veriyi yeniden şekillendir
        X_train = np.reshape(X_train, (n_samples, n_mels, time_steps))
        X_val = np.reshape(X_val, (X_val.shape[0], n_mels, time_steps))
    
    # Kanal boyutunu ekle
    X_train = X_train[..., np.newaxis]  # Şimdi (örnekler, n_mels, time_steps, 1) şeklinde
    X_val = X_val[..., np.newaxis]
    
    # Kontrol et ve şekli yazdır
    print(f"Reshape sonrası giriş boyutu: {X_train.shape[1:]}")
    
    input_shape = X_train.shape[1:]  # (n_mels, time_steps, 1)
    
    # Model oluştur - Spektrogramlar için Conv2D kullanılıyor
    model = tf.keras.Sequential([
        # İlk evrişimli blok
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # İkinci evrişimli blok
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Üçüncü evrişimli blok
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Düzleştirme ve yoğun katmanlar
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Modelin özetini göster
    model.summary()
    
    # Modeli derle
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Aşırı öğrenmeyi önlemek için erken durdurma
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Modeli eğit
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    # Doğrulama setinde değerlendir
    _, accuracy = model.evaluate(X_val, y_val)
    
    print(f"CNN Doğrulama Doğruluğu: {accuracy:.4f}")
    
    # Eğitim geçmişini çizdir
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Epoch')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Epoch')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
    
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'cnn_training_history.png'))
    
    return model, accuracy

# Ana fonksiyon
def main():
    # Veri yolları (gerekirse göreli yolları ayarla)
    base_dir = os.path.dirname(os.path.dirname(__file__)) # Proje köküne iki seviye yukarı çık
    train_metadata_path = os.path.join(base_dir, 'data', 'metadata', 'Metadata_Train.csv')
    test_metadata_path = os.path.join(base_dir, 'data', 'metadata', 'Metadata_Test.csv')
    train_dir = os.path.join(base_dir, 'data', 'raw', 'Train_submission')
    test_dir = os.path.join(base_dir, 'data', 'raw', 'Test_submission')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True) # results dizininin var olduğundan emin ol

    # Veriyi hazırla
    audio_paths, y_encoded, label_encoder = prepare_data(train_metadata_path, train_dir)
    
    # Özellik çıkarma
    feature_extractor = AudioFeatureExtractor()

    # Veriyi özellik çıkarmadan ÖNCE eğitim ve doğrulama setlerine ayır
    train_paths, val_paths, y_train_encoded, y_val_encoded = train_test_split(
        audio_paths, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # CNN için mel spektrogramlarını çıkar (Eğitim için augment=True, Doğrulama için augment=False)
    print("\nEğitim seti için mel spektrogramları çıkarılıyor (Veri Artırma ile)...") 
    X_train_spec, y_train_spec = feature_extractor.extract_mel_spectrograms(train_paths, y_train_encoded, augment=True) # augment=False -> augment=True
    
    print("\nDoğrulama seti için mel spektrogramları çıkarılıyor...")
    X_val_spec, y_val_spec = feature_extractor.extract_mel_spectrograms(val_paths, y_val_encoded, augment=False) # Doğrulama için augment=False
    
    # Spektrogramları eğitim ve doğrulama setlerine ayırma işlemini kaldır (zaten yapıldı)
    # X_train_spec, X_val_spec, y_train_spec, y_val_spec = train_test_split(...) # Bu satırı kaldır
    
    # CNN modelini eğit
    cnn_model, cnn_accuracy = train_cnn(
        X_train_spec, y_train_spec, X_val_spec, y_val_spec, len(label_encoder.classes_), results_dir) # results_dir'i ilet
    
    print(f"\nCNN modeli {cnn_accuracy:.4f} doğrulama doğruluğu ile eğitildi")
    
    # CNN modelini kaydet
    model_save_path = os.path.join(results_dir, 'best_model_cnn.h5')
    cnn_model.save(model_save_path)
    print(f"CNN modeli {model_save_path} olarak kaydedildi")
    
    # Etiket kodlayıcıyı betiğe göre veya results içinde kaydet
    label_encoder_path = os.path.join(results_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Etiket kodlayıcı {label_encoder_path} adresine kaydedildi")

    # Modeli test setinde test et
    test_metadata = pd.read_csv(test_metadata_path)
    test_paths = [os.path.join(test_dir, filename) for filename in test_metadata['FileName']]
    test_labels = label_encoder.transform(test_metadata['Class'])
    
    # Test verisi için mel spektrogramlarını çıkar
    print("\nTest seti için mel spektrogramları çıkarılıyor...")
    X_test_spec, _ = feature_extractor.extract_mel_spectrograms(test_paths, augment=False) # Test için augment=False
    X_test_spec = X_test_spec[..., np.newaxis] # Kanal boyutunu ekle
    
    # Tahmin için kaydedilen modeli yükle (isteğe bağlı, doğrudan cnn_model kullanılabilir)
    test_pred = np.argmax(cnn_model.predict(X_test_spec), axis=1)

    # Test sonuçlarını yazdır
    test_accuracy = accuracy_score(test_labels, test_pred)
    print(f"\nTest doğruluğu: {test_accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(test_labels, test_pred, target_names=label_encoder.classes_))
    
    # Karışıklık matrisini çizdir
    cm = confusion_matrix(test_labels, test_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Karışıklık Matrisi')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.tight_layout()
    plt.ylabel('Gerçek etiket')
    plt.xlabel('Tahmin edilen etiket')
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_save_path)
    
    print(f"Karışıklık matrisi görselleştirmesi {cm_save_path} olarak kaydedildi")

if __name__ == "__main__":
    main()
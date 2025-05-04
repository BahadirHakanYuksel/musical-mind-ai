import os
import sys
import numpy as np
import joblib
import tensorflow as tf
import librosa
import argparse

# Ana betikten özellik çıkarıcıyı içe aktar
from instrument_classifier import AudioFeatureExtractor

def load_cnn_model(model_path):
    """Kaydedilmiş bir CNN modelini (.h5 formatı) yükler."""
    print(f"CNN modeli yükleniyor: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"CNN modeli yüklenirken hata oluştu: {e}")
        sys.exit(1) # Model yükleme başarısız olursa çık

def predict_instrument(audio_path, model_path=None, label_encoder_path=None, results_dir='../../results'):
    """Verilen bir ses dosyası için enstrüman sınıfını bir CNN modeli kullanarak tahmin eder."""
    # Model yolunu belirle
    if model_path is None:
        model_path = os.path.join(results_dir, 'best_model_cnn.h5')
        if not os.path.exists(model_path):
             print(f"Varsayılan model yolu bulunamadı: {model_path}")
             # Yedek olarak results dizinindeki herhangi bir .h5 dosyasını ara
             h5_files = [f for f in os.listdir(results_dir) if f.endswith('.h5')]
             if h5_files:
                 model_path = os.path.join(results_dir, h5_files[0])
                 print(f"Yedek model kullanılıyor: {model_path}")
             else:
                 print(f"Hata: {results_dir} içinde CNN modeli (.h5) bulunamadı. Lütfen önce modeli eğitin veya --model kullanarak bir yol belirtin.")
                 sys.exit(1)

    # Etiket kodlayıcı yolunu belirle
    if label_encoder_path is None:
        label_encoder_path = os.path.join(results_dir, 'label_encoder.pkl')
        if not os.path.exists(label_encoder_path):
            print(f"Hata: Etiket kodlayıcı {label_encoder_path} adresinde bulunamadı. Lütfen eğitim sırasında kaydedildiğinden emin olun veya --label-encoder kullanarak bir yol belirtin.")
            sys.exit(1)

    print(f"Kullanılan model: {model_path}")
    print(f"Kullanılan etiket kodlayıcı: {label_encoder_path}")
    
    # CNN modelini yükle
    model = load_cnn_model(model_path)
    
    # Etiket kodlayıcıyı yükle
    try:
        label_encoder = joblib.load(label_encoder_path)
    except Exception as e:
        print(f"Etiket kodlayıcı yüklenirken hata oluştu: {e}")
        sys.exit(1)

    # Özellik çıkarıcı oluştur
    feature_extractor = AudioFeatureExtractor()
    
    # CNN için gereken sadece mel spektrogramını çıkar
    print(f"Mel spektrogramı çıkarılıyor: {audio_path}")
    try:
        # Basitleştirilmiş çıkarma: sesi yükle ve doğrudan spektrogramı al
        sr = feature_extractor.sample_rate
        duration = feature_extractor.duration
        samples = feature_extractor.samples
        n_mels = feature_extractor.n_mels
        n_fft = feature_extractor.n_fft
        hop_length = feature_extractor.hop_length

        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        if len(audio) < samples:
            audio = np.pad(audio, (0, samples - len(audio)), 'constant')
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec) # dB ölçeğine dönüştür

        X = mel_spec_db
        X = X[np.newaxis, ..., np.newaxis] # Batch ve kanal boyutlarını ekle: (1, n_mels, zaman_adımları, 1)

    except Exception as e:
        print(f"{audio_path} dosyasından özellik çıkarılamadı: {e}")
        return None

    # CNN modelini kullanarak tahmin yap
    print("Tahmin yapılıyor...")
    pred_proba = model.predict(X)[0]
    pred_class = np.argmax(pred_proba)
    
    # Sınıf olasılıklarını al
    class_probabilities = dict(zip(label_encoder.classes_, pred_proba))
    
    # Tahmin edilen sınıf adını al
    predicted_instrument = label_encoder.inverse_transform([pred_class])[0]
    
    print(f"\nTahmin Edilen Enstrüman: {predicted_instrument}")
    
    # Olasılıkları yazdır
    print("\nSınıf Olasılıkları:")
    for cls, prob in sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {prob:.4f}")
    
    return predicted_instrument

if __name__ == "__main__":
    # Argüman ayrıştırıcısını ayarla
    parser = argparse.ArgumentParser(description='Bir CNN modeli kullanarak ses dosyasından enstrüman türünü tahmin et.')
    parser.add_argument('audio_path', type=str, help='Ses dosyasının yolu.')
    parser.add_argument('--model', type=str, help='Kaydedilmiş CNN modelinin (.h5 dosyası) yolu. Varsayılan olarak results dizinindeki best_model_cnn.h5.', default=None)
    parser.add_argument('--label-encoder', type=str, help='Kaydedilmiş etiket kodlayıcının (.pkl dosyası) yolu. Varsayılan olarak results dizinindeki label_encoder.pkl.', default=None)
    parser.add_argument('--results-dir', type=str, help='Model ve kodlayıcı dosyalarını içeren dizinin yolu.', default='../../results') # Varsayılan göreli yol

    args = parser.parse_args()
    
    # Results dizininin var olduğundan veya doğru belirtildiğinden emin ol
    if not os.path.isdir(args.results_dir):
         # Betik konumundan göreli yolu çözmeyi dene
         script_dir = os.path.dirname(__file__)
         abs_results_dir = os.path.abspath(os.path.join(script_dir, args.results_dir))
         if os.path.isdir(abs_results_dir):
             args.results_dir = abs_results_dir
         else:
             print(f"Hata: Results dizini bulunamadı: {args.results_dir} (ayrıca {abs_results_dir} kontrol edildi)")
             sys.exit(1)


    # Ses dosyasının var olduğundan emin ol
    if not os.path.isfile(args.audio_path):
        print(f"Hata: Ses dosyası bulunamadı: {args.audio_path}")
        sys.exit(1)

    predict_instrument(args.audio_path, args.model, args.label_encoder, args.results_dir)
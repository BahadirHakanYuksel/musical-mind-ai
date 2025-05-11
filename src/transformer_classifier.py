import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Transformer için gerekli import'lar
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torchaudio
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

# Özellik çıkarma için sabitler
SAMPLE_RATE = 22050  # Standart örnekleme oranı
DURATION = 5  # Her ses dosyasının 5 saniyesini kullanacağız
N_MELS = 128  # Üretilecek mel bantlarının sayısı
N_FFT = 2048  # FFT penceresinin uzunluğu
HOP_LENGTH = 512  # Kareler arasındaki örnek sayısı

# Tekrarlanabilirlik için rastgele tohumları ayarla
np.random.seed(42)
torch.manual_seed(42)

# Custom Audio Dataset class
class AudioDataset(TorchDataset):
    def __init__(self, audio_paths, labels, feature_extractor, sample_rate=SAMPLE_RATE, max_length=DURATION):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # Get the feature extractor's required sample rate
        self.target_sample_rate = self.feature_extractor.sampling_rate
        print(f"Feature extractor requires sample rate: {self.target_sample_rate} Hz")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # Load audio at runtime instead of storing in memory
        try:
            try:
                audio, sr = sf.read(audio_path)
                # Mono'ya dönüştür (gerekirse)
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
            except Exception as e:
                # print(f"Soundfile ile yüklenemedi, librosa deneniyor: {audio_path}")
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_length)
            
            # Always resample to the target sample rate required by the feature extractor
            if sr != self.target_sample_rate:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.target_sample_rate)
                sr = self.target_sample_rate
            
            # Beklenen uzunlukta kırp veya doldur
            max_samples = int(self.target_sample_rate * self.max_length)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
            
            # Feature extraction and processing
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
                max_length=int(self.max_length * self.target_sample_rate)
            )
            
            # Get the first (and only) item from batch dim
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
            
            # Return inputs with label (if available)
            if self.labels is not None:
                return {
                    **inputs, 
                    "labels": torch.tensor(self.labels[idx])
                }
            return inputs
            
        except Exception as e:
            print(f"Hata oluştu ({audio_path}): {e}")
            # Return a zero tensor as fallback
            dummy_audio = np.zeros(int(self.target_sample_rate * self.max_length), dtype=np.float32)
            inputs = self.feature_extractor(
                dummy_audio, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt"
            )
            # Get the first (and only) item from batch dim
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
            
            # Return inputs with label (if available)
            if self.labels is not None:
                return {
                    **inputs, 
                    "labels": torch.tensor(self.labels[idx])
                }
            return inputs

class AudioTransformerModel:
    def __init__(self, num_labels, model_name="facebook/wav2vec2-base", sample_rate=SAMPLE_RATE):
        """
        Transformer tabanlı bir ses sınıflandırma modeli başlatır
        
        Parameters:
        - num_labels: Sınıf sayısı
        - model_name: Kullanılacak transformer model adı ('facebook/wav2vec2-base' veya AST modeli)
        - sample_rate: Örnekleme oranı
        """
        self.num_labels = num_labels
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cihaz: {self.device}")
        
        # Feature extractor'u başlat
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Transformer modelini başlat ve cihaza taşı
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # Önceden eğitilmiş modelin sınıf sayısı farklıysa görmezden gelir
        )
        self.model.to(self.device)
    
    def preprocess_audio(self, audio_paths, labels=None):
        """
        Ses dosyalarını transformer model için önişler ve veri sözlüğü oluşturur
        
        Parameters:
        - audio_paths: Ses dosyalarının yolları
        - labels: Etiketler (opsiyonel)
        
        Returns:
        - Veri sözlüğü
        """
        print("Ses dosyaları önişleniyor...")
        
        # Wav2Vec2 modeli için gerekli olan örnekleme oranı
        target_sample_rate = self.feature_extractor.sampling_rate
        
        # Önişleme için veri sözlüğünü başlat
        data_dict = {
            "audio": [],
            "input_values": [],
            "attention_mask": [],
        }
        
        if labels is not None:
            data_dict["labels"] = []
        
        # Her ses dosyası için işlem yap
        for i, audio_path in enumerate(audio_paths):
            try:
                # Ses dosyasını yükle
                try:
                    audio, sr = sf.read(audio_path)
                    # Mono'ya dönüştür (gerekirse)
                    if len(audio.shape) > 1 and audio.shape[1] > 1:
                        audio = np.mean(audio, axis=1)
                except Exception:
                    # print(f"Soundfile ile yüklenemedi, librosa deneniyor: {audio_path}")
                    audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=DURATION)
                
                # Wav2Vec2 modeli için gerekli olan örnekleme oranına resample et
                if sr != target_sample_rate:
                    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sample_rate)
                    sr = target_sample_rate
                
                # Beklenen uzunlukta kırp veya doldur
                max_samples = int(target_sample_rate * DURATION)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                elif len(audio) < max_samples:
                    audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
                
                data_dict["audio"].append(audio)
                
                if labels is not None:
                    data_dict["labels"].append(labels[i])
                
                # İlerleme durumunu göster
                if (i + 1) % 50 == 0:
                    print(f"{i+1}/{len(audio_paths)} dosya işlendi")
                    
            except Exception as e:
                print(f"Hata oluştu ({audio_path}): {e}")
                # Hata durumunda sıfır tensör kullan
                dummy_audio = np.zeros(int(target_sample_rate * DURATION), dtype=np.float32)
                data_dict["audio"].append(dummy_audio)
                
                if labels is not None:
                    data_dict["labels"].append(labels[i])
        
        return data_dict
    
    def prepare_datasets(self, audio_paths, labels):
        """Veri setini eğitim ve doğrulama setlerine ayırıp hazırlar"""
        print("Veri setleri hazırlanıyor...")
        
        # Eğitim ve doğrulama setlerine ayır
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            audio_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Audio path ve label sayılarını yazdır
        print(f"Eğitim seti: {len(train_paths)} dosya")
        print(f"Doğrulama seti: {len(val_paths)} dosya")
        
        # Veri setlerini oluştur
        train_dataset = AudioDataset(train_paths, train_labels, self.feature_extractor, 
                                   self.sample_rate, DURATION)
        val_dataset = AudioDataset(val_paths, val_labels, self.feature_extractor, 
                                 self.sample_rate, DURATION)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, num_epochs=10, batch_size=8):
        """Modeli eğitir"""
        print("Model eğitimi başlıyor...")
        
        # Optimizer ve kayıp fonksiyonunu ayarla
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()
        
        # DataLoader'ları oluştur
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Eğitim
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            progress_interval = max(1, len(train_loader) // 10)  # Progress update interval
            
            for i, batch in enumerate(train_loader):
                # Forward pass
                optimizer.zero_grad()
                
                # Move inputs to device
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask
                ) if attention_mask is not None else self.model(input_values=input_values)
                
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Print progress
                if (i + 1) % progress_interval == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {i+1}/{len(train_loader)}")
            
            # Doğrulama
            self.model.eval()
            total_val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move inputs to device
                    input_values = batch["input_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    outputs = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    ) if attention_mask is not None else self.model(input_values=input_values)
                    
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    
                    total_val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # İstatistikler
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f} - "
                  f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save model checkpoint after each epoch
            checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'results', 'transformer', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_acc_{val_accuracy:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, checkpoint_path)
            print(f"Checkpoint kaydedildi: {checkpoint_path}")
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
    
    def evaluate(self, test_paths, test_labels, batch_size=8):
        """Test veri seti üzerinde modeli değerlendirir"""
        print("Model değerlendiriliyor...")
        
        self.model.eval()
        
        # Create test dataset
        test_dataset = AudioDataset(test_paths, test_labels, self.feature_extractor, 
                                   self.sample_rate, DURATION)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move inputs to device
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask
                ) if attention_mask is not None else self.model(input_values=input_values)
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        return all_preds, all_labels
    
    def save_model(self, save_dir):
        """Modeli kaydeder"""
        print(f"Model {save_dir} konumuna kaydediliyor...")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)
    
    def load_model(self, load_dir):
        """Kaydedilmiş bir modeli yükler"""
        print(f"Model {load_dir} konumundan yükleniyor...")
        self.model = AutoModelForAudioClassification.from_pretrained(load_dir)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(load_dir)
        self.model.to(self.device)
        
    def plot_training_history(self, history, save_path=None):
        """Eğitim sürecinin kayıp ve doğruluk grafiklerini çizer"""
        plt.figure(figsize=(12, 4))
        
        # Kayıp grafiği
        plt.subplot(1, 2, 1)
        plt.plot(history["train_losses"], label="Train")
        plt.plot(history["val_losses"], label="Validation")
        plt.title("Loss During Training")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        
        # Doğruluk grafiği
        plt.subplot(1, 2, 2)
        plt.plot(history["val_accuracies"], label="Validation")
        plt.title("Accuracy During Training")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Eğitim grafiği kaydedildi: {save_path}")
        
        plt.close()

# Ana fonksiyon
def main():
    # Veri yolları (göreli yolları ayarla)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_metadata_path = os.path.join(base_dir, 'data', 'metadata', 'Metadata_Train.csv')
    test_metadata_path = os.path.join(base_dir, 'data', 'metadata', 'Metadata_Test.csv')
    train_dir = os.path.join(base_dir, 'data', 'raw', 'Train_submission')
    test_dir = os.path.join(base_dir, 'data', 'raw', 'Test_submission')
    results_dir = os.path.join(base_dir, 'results')
    transformer_results_dir = os.path.join(results_dir, 'transformer')
    transformer_model_dir = os.path.join(transformer_results_dir, 'model')
    os.makedirs(transformer_results_dir, exist_ok=True)

    # Veriyi hazırla
    print("Veri hazırlanıyor...")
    train_metadata = pd.read_csv(train_metadata_path)
    test_metadata = pd.read_csv(test_metadata_path)
    
    train_audio_paths = [os.path.join(train_dir, filename) for filename in train_metadata['FileName']]
    train_labels = train_metadata['Class'].values
    
    test_audio_paths = [os.path.join(test_dir, filename) for filename in test_metadata['FileName']]
    test_labels = test_metadata['Class'].values
    
    # Etiketleri kodla
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_labels)
    y_test_encoded = label_encoder.transform(test_labels)
    
    num_labels = len(label_encoder.classes_)
    print(f"Sınıf sayısı: {num_labels}")
    
    # Etiket kodlayıcıyı kaydet
    joblib.dump(label_encoder, os.path.join(transformer_results_dir, 'label_encoder.pkl'))
    
    # Hyperparameters
    batch_size = 8  # Reduce batch size to save memory
    num_epochs = 3  # Reduce epochs for faster training
    
    # Transformer modelini oluştur
    model_name = "facebook/wav2vec2-base"  # veya "MIT/ast-finetuned-audioset-10-10-0.4593" gibi bir AST modeli
    audio_transformer = AudioTransformerModel(num_labels=num_labels, model_name=model_name)
    
    # Verileri hazırla
    train_dataset, val_dataset = audio_transformer.prepare_datasets(train_audio_paths, y_train_encoded)
    
    # Modeli eğit
    training_history = audio_transformer.train(
        train_dataset, 
        val_dataset, 
        num_epochs=num_epochs, 
        batch_size=batch_size
    )
    
    # Eğitim grafiklerini çiz
    audio_transformer.plot_training_history(
        training_history, 
        save_path=os.path.join(transformer_results_dir, 'transformer_training_history.png')
    )
    
    # Test veri seti üzerinde değerlendir
    predictions, ground_truth = audio_transformer.evaluate(test_audio_paths, y_test_encoded, batch_size=batch_size)
    
    # Karışıklık matrisini çiz
    cm = confusion_matrix(ground_truth, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
    plt.yticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(transformer_results_dir, 'transformer_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Karışıklık matrisi kaydedildi: {cm_path}")
    plt.close()
    
    # Modeli kaydet
    audio_transformer.save_model(transformer_model_dir)
    print(f"Transformer modeli başarıyla kaydedildi: {transformer_model_dir}")

if __name__ == "__main__":
    main()
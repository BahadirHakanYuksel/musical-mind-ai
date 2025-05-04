<!-- filepath: /home/erguvenburak/Belgeler/ANN/musical-mind-ai/README_tr.md -->

# CNN Kullanarak Müzik Enstrümanı Sınıflandırması

Bu proje, Mel Spektrogramlarını kullanarak ses dosyalarından müzik enstrümanlarını sınıflandırmak için Evrişimli Sinir Ağı (CNN) tabanlı bir sistem uygular. Model, 3 farklı enstrüman türünü tanımlamak üzere eğitilmiştir:

- Gitar
- Davul
- Piyano

[Click here for English README](README.md)

## Proje Yapısı

- `src/instrument_classifier.py`: Mel Spektrogramlarını çıkarmak, veri artırma ile CNN modelini eğitmek, performansı değerlendirmek ve en iyi modeli kaydetmek için ana betik.
- `src/predict_instrument.py`: Eğitilmiş CNN modelini kullanarak yeni bir ses dosyasının sınıfını tahmin etmek için betik.
- `src/activation_comparison.py`: CNN model mimarisi içinde farklı aktivasyon fonksiyonlarını karşılaştırmak için betik.
- `requirements.txt`: Paket bağımlılıkları.
- `data/metadata/Metadata_Train.csv`: Eğitim verisi dosya adlarını ve karşılık gelen enstrüman sınıflarını içeren CSV dosyası.
- `data/metadata/Metadata_Test.csv`: Test verisi dosya adlarını ve karşılık gelen enstrüman sınıflarını içeren CSV dosyası.
- `data/raw/Train_submission/`: Eğitim ses dosyalarını içeren dizin.
- `data/raw/Test_submission/`: Test ses dosyalarını içeren dizin.
- `results/`: Eğitilmiş modellerin, değerlendirme sonuçlarının ve görselleştirmelerin kaydedildiği dizin.

## Çıkarılan Özellikler

Sistem, öncelikli olarak ses dosyalarından türetilen **Mel Spektrogramlarını** CNN modeli için girdi olarak kullanır. `instrument_classifier.py` betiği ayrıca MFCC'ler, spektral özellikler, ZCR, RMS, Kroma, tempo ve harmonik/vurmalı bileşenler gibi diğer özellikleri çıkarmak için kod içerir, ancak bunlar mevcut ana betikteki son CNN modeli için doğrudan girdi olarak kullanılmaz.

- **Mel Spektrogramları**: CNN'ler için uygun zaman-frekans gösterimi.

Modelin sağlamlığını artırmak için eğitim sırasında veri artırma teknikleri (gürültü ekleme, perde kaydırma, zaman uzatma) uygulanır.

## Uygulanan Model

Sistemin çekirdeği, Mel Spektrogram görüntülerini işlemek üzere tasarlanmış bir **Evrişimli Sinir Ağı (CNN)**'dır. Mimari şunları içerir:

- Özellik çıkarımı için Batch Normalization ve Dropout içeren Conv2D katmanları.
- Örneklemeyi azaltmak için MaxPooling katmanları.
- Yoğun katmanlara geçiş yapmak için Flatten katmanı.
- Sınıflandırma için Dense katmanları.
- Çok sınıflı olasılık dağılımı için çıktı katmanında Softmax aktivasyonu.

Model, Adam optimize ediciyi ve sparse categorical crossentropy kayıp fonksiyonunu kullanır. Aşırı öğrenmeyi önlemek için erken durdurma (early stopping) kullanılır.

## Aktivasyon Fonksiyonları Karşılaştırması

`src/activation_comparison.py` betiği, performans üzerindeki etkilerini belirlemek için farklı aktivasyon fonksiyonları kullanarak CNN modelini eğitir ve değerlendirir. Karşılaştırılan fonksiyonlar şunlardır:

- ReLU
- Tanh
- Sigmoid
- ELU
- SELU
- Swish

Sonuçlar (doğruluk ve eğitim süresi) `cnn_activation_results.csv` dosyasına kaydedilir ve `activation_comparison.png` dosyasında görselleştirilir.

## Test Sonuçları ve Görselleştirmeler

Eğitim süreci ve değerlendirme, `results/` dizininde (ve karşılaştırma sonuçları için kök dizinde) kaydedilen birkaç çıktı dosyası üretir:

### Karışıklık Matrisi

![Karışıklık Matrisi](results/confusion_matrix.png)

En iyi CNN modelinin test veri kümesindeki performansını gösterir, sınıf başına doğru ve yanlış tahminleri detaylandırır.

### CNN Eğitim Geçmişi

![CNN Eğitim Geçmişi](results/cnn_training_history.png)

CNN modeli için epoch'lar boyunca eğitim ve doğrulama doğruluğunu/kaybını çizer.

### Aktivasyon Fonksiyonu Karşılaştırması

![Aktivasyon Karşılaştırması](activation_comparison.png)

CNN'deki farklı aktivasyon fonksiyonları için doğrulama doğruluğu ve eğitim süresinin görsel karşılaştırması.

## Kullanım

### CNN Modelini Eğitme

```bash
python src/instrument_classifier.py
```

Bu betik şunları yapacaktır:

- Eğitim meta verilerini yükler.
- Eğitim verileri için veri artırma ile Mel Spektrogramlarını çıkarır.
- Doğrulama verileri için Mel Spektrogramlarını çıkarır.
- CNN modelini eğitir.
- Modeli test setinde değerlendirir.
- En iyi modeli (`results/best_model_cnn.h5`), etiket kodlayıcıyı (`results/label_encoder.pkl`), karışıklık matrisini (`results/confusion_matrix.png`) ve eğitim geçmişini (`results/cnn_training_history.png`) kaydeder.

### Eğitilmiş Model ile Tahmin Yapma

```bash
python src/predict_instrument.py path/to/your/audiofile.wav [--model results/best_model_cnn.h5] [--encoder results/label_encoder.pkl]
```

Eğitilmiş modeli yükler ve belirtilen ses dosyası için enstrüman sınıfını tahmin eder.

### Aktivasyon Fonksiyonlarını Karşılaştırma

```bash
python src/activation_comparison.py
```

Bu betik şunları yapacaktır:

- Eğitim verilerini yükler.
- Mel Spektrogramlarını çıkarır.
- Farklı aktivasyon fonksiyonlarıyla CNN modellerini eğitir ve değerlendirir.
- Karşılaştırma sonuçlarını `cnn_activation_results.csv` ve `activation_comparison.png` dosyalarına kaydeder.

## Kurulum

1.  Depoyu klonlayın (henüz yapmadıysanız).
2.  Bir sanal ortam oluşturun ve etkinleştirin (önerilir):
    ```bash
    python -m venv env
    source env/bin/activate  # Windows'ta `env\Scripts\activate` kullanın
    ```
3.  Bağımlılıkları yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
    _Not: GPU hızlandırması istiyorsanız `tensorflow[and-cuda]`, uygun NVIDIA sürücülerinin ve CUDA araç setinin kurulu olmasını gerektirir._

## Üretilen Sonuç Dosyaları

- `results/best_model_cnn.h5`: Eğitilmiş CNN modeli.
- `results/label_encoder.pkl`: Enstrüman sınıfları için uydurulmuş etiket kodlayıcı.
- `results/confusion_matrix.png`: Test seti performansının görselleştirilmesi.
- `results/cnn_training_history.png`: Eğitim/doğrulama doğruluğu ve kaybının grafiği.
- `activation_comparison.png`: Aktivasyon fonksiyonu karşılaştırma sonuçlarının görselleştirilmesi.
- `cnn_activation_results.csv`: Aktivasyon fonksiyonu performansını karşılaştıran tablo.

# Klasik Görüntü İşleme ile Mikroskobik Görüntülerde Hücre Segmentasyonu, Sayımı ve Zamansal Analizi

Görüntü İşleme (Yüksek Lisans) ara sınav projesi · Öğrenci No: **254312024**

Fluo-N2DH-GOWT1 (Cell Tracking Challenge) floresan mikroskopi sekanslarında yalnızca **klasik** görüntü işleme
yöntemleriyle (yoğunluk dönüşümü, uzamsal filtreleme, eşikleme, morfoloji, distance transform + watershed)
hücre segmentasyonu ve kare bazında hücre sayımı yapılır. Yöntem ve parametreler geliştirme kümesinde kontrollü
deneylerle seçilir, doğrulama kümesinde aday pipeline'lar karşılaştırılır ve kilitlenen nihai pipeline bağımsız
test kareleri üzerinde değerlendirilir. Derin öğrenme veya önceden eğitilmiş model kullanılmamıştır.

## Veri seti
* Sayfa: https://celltrackingchallenge.net/2d-datasets/
* İndirme (eğitim arşivi): https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip
* Arşivi `data/` klasörüne açın: `data/Fluo-N2DH-GOWT1/01`, `01_GT/TRA`, `02`, `02_GT/TRA` …
  (Zip dosyası `data/`, proje klasörü veya bir üst klasörde ise notebook onu otomatik açar.)
* Veri setine ait görüntüler ve maskeler bu repository'de **paylaşılmaz** (`.gitignore`).

## Tekrarlanabilirlik
* Random seed: **14050** (= öğrenci numarasının son 5 hanesi 12024 + 2026), `numpy.random.default_rng`.
* Seçim: Sequence 01'den 5 doğrulama + 10 geliştirme karesi (aralarında en az 5 kare boşluk),
  Sequence 02'den ardışık 5'er karelik, aralarında en az 5 kare boşluk olan 3 test bloğu.
  Koşul sağlanamazsa aynı RNG akışından deterministik yeniden çekim yapılır (`cellcount/selection.py`).
* Referans sayısı: `XX_GT/TRA/man_trackTTT.tif` içindeki 0 dışındaki farklı etiket sayısı.

## Kurulum
```bash
pip install -r requirements.txt
```
Anaconda kullanıcıları için ek paket gerekmez (numpy, scipy, opencv, scikit-image, matplotlib, pandas yeterlidir).
TIFF'ler LZW sıkıştırmalıdır; `imagecodecs` gerektirmemek için OpenCV ile okunur.

Geliştirmede kullanılan ortam: Python 3.13.9 (Anaconda), Windows; sürümler `requirements.txt` ve
`final_pipeline.json → software` alanında kayıtlıdır.

**Jupyter (yerel):** Proje klasöründe `jupyter notebook` → `main.ipynb` → *Run All*.

**Google Colab:** İlk hücreden önce aşağıdakileri çalıştırın (notebook'ların proje klasöründe çalışması gerekir):
```python
!git clone https://github.com/mehmetsencodes-ship-it/MehmetSen_254312024_GoruntuIsleme_Vize.git
%cd MehmetSen_254312024_GoruntuIsleme_Vize
!pip install -q -r requirements.txt
!wget -q -P data https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip
```
Zip dosyası `data/` altında bulunduğunda notebook onu otomatik açar; başka bir yol düzenlemesi gerekmez.

## Çalıştırma sırası
1. **`main.ipynb`** (Bölüm 2–12): veri incelemesi → kare seçimi → dev/val referansları → baseline →
   kontrollü deneyler → 3 aday pipeline → doğrulama → nihai pipeline kilitleme ve test tahmini.
   Test referansları bu notebook'ta okunmaz.
2. **Ayrı commit:** `final_pipeline.json`, `test_predictions_locked.csv`, `test_lock.json`
   ```bash
   git add final_pipeline.json test_predictions_locked.csv test_lock.json
   git commit -m "Lock final pipeline and test predictions (before reading test references)"
   ```
3. **`02_test_evaluation.ipynb`** (Bölüm 13–14): kilit dosyalarının SHA-256 doğrulaması → test referanslarının
   okunması → metrikler, zamansal analiz, hata analizi. Ardından sonuçlar ikinci bir commit ile yüklenir.

Kilit dosyaları mevcutken `main.ipynb` yeniden çalıştırılırsa dosyaların üzerine **yazılmaz**; yalnızca aynı
konfigürasyonun ve aynı test tahminlerinin yeniden üretildiği doğrulanır.

Kod yeniden çalıştırıldığında veri bölünmesi, referans sayıları, tüm tahminler ve hata metrikleri birebir aynı
çıkar; yalnızca işlem süresi sütunları donanıma ve sistem yüküne bağlı olarak küçük farklılıklar gösterebilir.

GitHub'a yüklemeden önce notebook çıktılarını temizleyin: `jupyter nbconvert --clear-output --inplace *.ipynb`

## Nihai pipeline (Pipeline B)
`Histogram Equalization → Mean filtre 5×5 → Otsu → Opening (elips 3×3) → Fill holes →
Distance transform + marker tabanlı watershed (peak_local_max, min_distance=35) → min alan 400 px → etiket sayısı`

Baseline: `Gri görüntü → Otsu → 8-bağlı bileşen etiketleme → min alan 50 px → sayım`.

## Kod yapısı
| Dosya | İçerik |
|---|---|
| `cellcount/io_utils.py` | Veri okuma, TRA referans sayımı |
| `cellcount/selection.py` | Seed tabanlı kare seçimi ve kural kontrolü |
| `cellcount/ops.py` | Yoğunluk dönüşümleri, filtreler, segmentasyon, morfoloji, watershed |
| `cellcount/pipeline.py` | JSON ile tanımlı pipeline, baseline, değerlendirme |
| `cellcount/experiments.py` | Kontrollü deney protokolü (aşamalı tek-faktör araması) |
| `cellcount/metrics.py` | MAE, RMSE, Bias, Exact-count Accuracy |
| `cellcount/viz.py` | Şekil yardımcıları, akış diyagramı |

## Üretilen dosyalar
| Dosya | Açıklama |
|---|---|
| `selected_frames.csv` | Seed, veri grubu, sekans, kare no, dosya adı, test bloğu |
| `dataset_stats.csv` | Görüntü istatistikleri (ön inceleme) |
| `cell_shape_stats.csv` | Sequence 01 SEG anotasyonlarından çekirdek alan, çap, eksantriklik, katılık, parlaklık |
| `reference_counts_devval.csv` | Geliştirme/doğrulama referans hücre sayıları |
| `experiments.csv` | Baseline (B01–B06) ve kontrollü deneyler (E01–E52): yöntem, parametre, dev MAE/RMSE/Bias, süre |
| `validation_results.csv` | Baseline + 3 aday pipeline'ın geliştirme ve doğrulama sonuçları, süre, karmaşıklık ve seçim |
| `final_pipeline.json` | Nihai pipeline + baseline konfigürasyonu, seçim kuralı, yazılım sürümleri |
| `test_predictions_locked.csv` | Test referansları okunmadan kaydedilen baseline ve nihai tahminler |
| `test_lock.json` | Kilit dosyalarının SHA-256 özetleri |
| `test_results.csv` | Test kareleri: gerçek sayı, tahminler, hatalar |
| `test_summary.csv`, `test_per_block.csv` | Özet test metrikleri (genel ve blok bazında) |
| `test_temporal_stats.csv` | Ardışık kare sayım değişim istatistikleri |
| `test_error_analysis.csv` | Kaçırılan / eşleşmeyen / birleşmiş tespit analizi |
| `figures/*.png` | Veri görüntüsü içermeyen grafikler (parametre etkisi, akış diyagramı, zamansal grafik) |
| `figures/private/*.png` | Veri görüntüsü içeren şekiller – yalnızca raporda kullanılır, GitHub'a yüklenmez |

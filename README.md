**Giriş ve Senaryo Tanımı**
- **Özet**: Bu proje, 2B ortamda otonom mobil robotun hedefe ulaşması ve engelden kaçınmasını simüle eder. Robot, LiDAR, enkoder ve IMU sensörleri kullanılarak lokalize edilir; sensör verileri EKF ile füzyonlanır ve non-holonomic hareket modeli kullanılır.
- **Senaryo**: Başlangıç ve hedef konumları verilmiş, rastgele dairesel engeller yerleştirilmiş bir ortamda robotun planlanan yolunu ve gerçek izlemesini değerlendirme.

**Kullanılan Yöntemler**
- **LiDAR Simülasyonu**: Işın (ray) - çember kesişimi ile 2D LiDAR ham verisi üretilir; mesafe eşikleme uygulanır.
- **Kümeleme**: Ardışık mesafe tabanlı basit kümeleme ile LiDAR noktalarından engel merkezleri çıkartılır.
- **Sensörler ve Ölçümler**: Enkoder (dead reckoning), IMU (yaw ölçümü) ve LiDAR.
- **Sensör Füzyonu**: Enkoder + IMU ölçüleri EKF (Genişletilmiş Kalman Filtresi) ile birleştirilir; ölçüm modeli IMU için açıyı güncelleyen basit bir H matrisi içerir.
- **Kontrol ve Kaçınma**: Non-holonomic hareket modeli ve potansiyel alan tabanlı (Potential Field) reaktif kontrol ile engelden kaçınma ve hedefe yönelme sağlanır.

**Sonuçlar ve Grafikler**
- **Çıktılar**: Simülasyon görselleri tek bir çok sayfalı PDF olarak `results.pdf` içinde saklanır.
- **Her sayfa**: 1) Ortam ve yollar (planlanan, gerçek, EKF, DR), 2) LiDAR ham noktaları ve filtrelenmiş küme merkezleri, 3) x(t) karşılaştırması, 4) y(t) karşılaştırması, 5) θ(t) karşılaştırması, 6) Hata analizi ve RMSE.
- **Dosyalar**: Ana kod `ödev.py` simülasyonu üretir ve `results.pdf` kaydeder.

**Hata Analizi ve Kısa Tartışma**
- **Hata metrikleri**: Konum hatası zaman serisi ve RMSE değerleri hesaplanır; Dead Reckoning (DR) ile EKF karşılaştırılır.
- **Gözlemler**: Tipik olarak EKF, enkoder yalnız başına yapılan dead reckoning'e göre daha düşük RMSE sağlar çünkü IMU/ölçü güncellemeleri sapmayı düzeltir.
- **Sınırlamalar**: Model ve sensör parametreleri (gürültü, sensör menzili, kümeleme eşikleri) sonucu etkiler; LiDAR için görüş engellemeleri/çakışmalar hata yaratabilir. Gerçek robotta gri alanlar (senkronizasyon, zaman gecikmeleri) ek zorluk oluşturur.

**Kaynaklar ve Yapay Zeka Kullanım Beyanı**
- **Kaynaklar (örnek)**: Kalman filtresi ve genişletilmiş Kalman filtresi literatürü; potansiyel alan tabanlı reaktif kontrol kaynakları.
- **Yapay Zeka Kullanım Beyanı**: Bu proje sırasında kod düzenlemeleri, görselleştirme iyileştirmeleri ve README içeriğinin oluşturulmasında bir yapay zeka asistanından (kod yardımcısı) destek alınmıştır. Son kararlar ve doğrulamalar kullanıcı tarafından yapılmıştır.

**Çalıştırma ve Gereksinimler (kısa)**
- **Gereksinimler**: `numpy`, `matplotlib`, `python-docx` vb. (örnek yükleme: `pip install numpy matplotlib python-docx`).
- **Çalıştırma**: Proje kökünde aşağıdaki komutla simülasyonu çalıştırabilirsiniz:

```bash
python ödev.py
```

- Üretilen grafikler `results.pdf` içinde toplanacaktır.

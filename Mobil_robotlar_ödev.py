import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. SİMÜLASYON VE ORTAM PARAMETRELERİ
# ==========================================
DT = 0.1  # Zaman adımı (s)
SIM_TIME = 50.0  # Maksimum simülasyon süresi

# Başlangıç ve Hedef Noktaları
START = np.array([0.0, 0.0])
GOAL = np.array([10.0, 10.0])

# En az 10 Engel - Rastgele konumlarda oluşturulacak
# Engel sayısı
NUM_OBSTACLES = 10

def generate_random_obstacles(num_obstacles, start, goal, min_dist_to_line=0.5):
    """Rastgele engel konumları oluştur (başlangıç-hedef hattından uzak)"""
    obstacles = []
    np.random.seed()  # Her seferinde farklı rastgele sayılar
    
    while len(obstacles) < num_obstacles:
        # 0.5 ile 10 arasında rastgele x, y konumları
        ox = np.random.uniform(0.5, 9.5)
        oy = np.random.uniform(0.5, 9.5)
        orad = np.random.uniform(0.4, 0.7)  # Yarıçap da rastgele
        
        # Başlangıç ve hedef noktalarına çok yakın olmasını engelle
        dist_to_start = math.hypot(ox - start[0], oy - start[1])
        dist_to_goal = math.hypot(ox - goal[0], oy - goal[1])
        
        if dist_to_start > 1.5 and dist_to_goal > 1.5:
            obstacles.append((ox, oy, orad))
    
    return obstacles

# Gürültü Parametreleri (Sensör Gürültüsü Koşulları) - [cite: 39]
Q = np.diag([0.1, 0.1, np.deg2rad(1.0)]) ** 2  # Süreç gürültüsü (Dead Reckoning)
R = np.diag([0.2, 0.2]) ** 2  # Ölçüm gürültüsü (LiDAR / Konum)

# ==========================================
# 2. NON-HOLONOMİK ROBOT VE EKF (LOKALİZASYON)
# ==========================================
def motion_model(x, u):
    """ Non-holonomik kinematik model (Diferansiyel Sürüş)  """
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])
    return F @ x + B @ u

def observation_model(x):
    """ LiDAR/Sensör füzyonu ile konum ölçüm modeli [cite: 68] """
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])
    return H @ x

def jacob_f(x, u):
    """ Jacobian of Motion Model """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw)],
        [0.0, 0.0, 1.0]])
    return jF

def ekf_estimation(xEst, PEst, z, u):
    """ Genişletilmiş Kalman Filtresi [cite: 62] """
    # Tahmin (Predict) - Dead Reckoning
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    # Güncelleme (Update) - Sensör Füzyonu
    H = np.array([[1, 0, 0], [0, 1, 0]])
    zPred = observation_model(xPred)
    y = z - zPred
    S = H @ PPred @ H.T + R
    K = PPred @ H.T @ np.linalg.inv(S)
    
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ H) @ PPred
    return xEst, PEst

# ==========================================
# 3. YAPAY POTANSİYEL ALAN (APF) NAVİGASYON
# ==========================================
def potential_field_control(x, goal, obstacles):
    """ Hedefe çekim, engellerden agresif itim (Codex Tuning) """
    KP = 1.1     # Çekim katsayısı
    ETA = 25.0   # İtim katsayısı (Güçlendirildi)
    D_OBS = 2.0  # Engel etki mesafesi (Artırıldı)

    # Çekim Kuvveti (Hedefe doğru)
    fx = KP * (goal[0] - x[0, 0])
    fy = KP * (goal[1] - x[1, 0])

    # İtim Kuvveti (Engellerden kaçınma)
    for (ox, oy, orad) in obstacles:
        dist = math.hypot(x[0, 0] - ox, x[1, 0] - oy)
        
        # Engel etki alanındaysa ve üst üste (dist=0) değilse
        if dist < D_OBS and dist > 0.01: 
            dq = dist - orad
            
            # Tam çarpışma anında sıfıra bölmeyi (singularity) engelle
            if dq < 0.01: 
                dq = 0.01 
            
            # Codex'in önerdiği basitleştirilmiş ve sert itim formülü
            force = ETA / (dq ** 2)
            
            # YÖN VEKTÖRÜ: (x - ox) uzaklaştırıcı vektördür. 
            # += ile eklenmelidir.
            fx += force * (x[0, 0] - ox) / dist
            fy += force * (x[1, 0] - oy) / dist

    # Hedef açı ve lineer hız hesabı (Non-holonomik dönüşüm)
    target_yaw = math.atan2(fy, fx)
    yaw_error = target_yaw - x[2, 0]
    
    # Açıyı -pi ile pi arasına normalize et
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
    
    v = math.hypot(fx, fy)
    v = np.clip(v, 0.0, 1.5)  # Maksimum hız sınırı
    w = 2.0 * yaw_error       # Basit oransal açısal hız kontrolcüsü
    w = np.clip(w, -1.5, 1.5) # Açısal hız sınırını biraz artırarak daha keskin manevra sağlandı
    
    return np.array([[v], [w]])

# ==========================================
# 4. ANA SİMÜLASYON DÖNGÜSÜ
# ==========================================
def main():
    # Rastgele engelleri oluştur
    OBSTACLES = generate_random_obstacles(NUM_OBSTACLES, START, GOAL)
    
    time = 0.0
    
    # Durum Vektörleri: [x, y, theta]
    xTrue = np.zeros((3, 1))
    xTrue[0, 0], xTrue[1, 0] = START[0], START[1]
    
    xDR = np.copy(xTrue)  # Dead Reckoning
    xEst = np.copy(xTrue) # EKF Tahmini
    PEst = np.eye(3)
    
    # Geçmiş verileri kaydetmek için (Grafikler için)
    hxTrue, hxDR, hxEst, hError = [], [], [], []

    while time < SIM_TIME:
        time += DT
        
        # Kontrol girdisini hesapla (APF)
        u = potential_field_control(xEst, GOAL, OBSTACLES)
        
        # Gerçek durumu simüle et (Süreç gürültüsü ile)
        u_true = u + np.random.randn(2, 1) * np.array([[0.05], [0.05]])
        xTrue = motion_model(xTrue, u_true)
        
        # Dead Reckoning hesapla
        xDR = motion_model(xDR, u)
        
        # Sensör Ölçümü Simülasyonu (LiDAR konum gürültüsü eklenmiş)
        z = observation_model(xTrue) + np.random.randn(2, 1) * np.array([[np.sqrt(R[0,0])], [np.sqrt(R[1,1])]])
        
        # EKF ile Lokalizasyon [cite: 52]
        xEst, PEst = ekf_estimation(xEst, PEst, z, u)
        
        # Hata Analizi (RMSE hesabı için mesafe farkı) [cite: 107]
        error = math.hypot(xTrue[0, 0] - xEst[0, 0], xTrue[1, 0] - xEst[1, 0])
        
        # Geçmişi kaydet
        hxTrue.append(xTrue[:, 0].tolist())
        hxDR.append(xDR[:, 0].tolist())
        hxEst.append(xEst[:, 0].tolist())
        hError.append(error)
        
        # Hedefe varış kontrolü
        if math.hypot(xTrue[0, 0] - GOAL[0], xTrue[1, 0] - GOAL[1]) < 0.5:
            print("Hedefe ulaşıldı!")
            break

    hxTrue = np.array(hxTrue)
    hxDR = np.array(hxDR)
    hxEst = np.array(hxEst)
    
    # ==========================================
    # 5. GÖRSELLEŞTİRME VE GRAFİKLER
    # ==========================================
    plt.figure(figsize=(12, 5))
    
    # 1. Ortam ve Yol Haritası [cite: 86, 91]
    plt.subplot(1, 2, 1)
    plt.plot(START[0], START[1], "go", markersize=10, label="Başlangıç")
    plt.plot(GOAL[0], GOAL[1], "r*", markersize=10, label="Hedef")
    
    for (ox, oy, orad) in OBSTACLES:
        circle = plt.Circle((ox, oy), orad, color='gray')
        plt.gca().add_patch(circle)
        
    plt.plot(hxTrue[:, 0], hxTrue[:, 1], "-b", label="Gerçek Yol")
    plt.plot(hxDR[:, 0], hxDR[:, 1], "-k", label="Dead Reckoning")
    plt.plot(hxEst[:, 0], hxEst[:, 1], "-r", label="EKF Tahmini")
    
    plt.title("2B Otonom Navigasyon ve Lokalizasyon")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    
    # 2. Hata Analizi (Zaman Boyunca Konum Hatası) [cite: 105, 106]
    plt.subplot(1, 2, 2)
    time_arr = np.arange(0, len(hError) * DT, DT)
    plt.plot(time_arr, hError, "-g", label="Konum Hatası")
    
    rmse = np.sqrt(np.mean(np.array(hError)**2))
    plt.title(f"Zaman Boyunca Hata (RMSE: {rmse:.3f} m)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Hata (m)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Oluşturulan engellerin konumlarını yazdır
    print("\n" + "="*50)
    print("Rastgele Oluşturulan Engel Konumları:")
    print("="*50)
    for i, (ox, oy, orad) in enumerate(OBSTACLES, 1):
        print(f"Engel {i}: Merkez=({ox:.2f}, {oy:.2f}), Yarıçap={orad:.2f}")

if __name__ == '__main__':
    main()
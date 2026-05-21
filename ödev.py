import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. SİMÜLASYON VE ORTAM PARAMETRELERİ
# ==========================================
DT = 0.1  # Zaman adımı (s)
SIM_TIME = 50.0  # Maksimum simülasyon süresi

# Robot ve algılama parametreleri
WHEEL_BASE = 0.5
MAX_LIDAR_RANGE = 6.0
LIDAR_NUM_BEAMS = 72
LIDAR_THRESHOLD = 5.5
CLUSTER_DISTANCE_THRESHOLD = 0.45
IMU_YAW_NOISE_STD = np.deg2rad(2.0)
ENCODER_V_NOISE_STD = 0.03

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
R_IMU = np.array([[IMU_YAW_NOISE_STD ** 2]])


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def transform_robot_to_world(x, y, yaw, points_robot):
    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)]
    ])
    translation = np.array([[x], [y]])
    return (rotation @ points_robot.T).T + translation.T


def ray_circle_intersection(robot_pose, beam_angle, obstacle, max_range=MAX_LIDAR_RANGE):
    rx, ry, _ = robot_pose.flatten()
    ox, oy, radius = obstacle

    dx = math.cos(beam_angle)
    dy = math.sin(beam_angle)
    fx = rx - ox
    fy = ry - oy

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    discriminant = b * b - 4.0 * a * c

    if discriminant < 0.0:
        return None

    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2.0 * a)
    t2 = (-b + sqrt_discriminant) / (2.0 * a)

    valid = [t for t in (t1, t2) if 0.0 < t <= max_range]
    if not valid:
        return None

    return min(valid)


def simulate_lidar_scan(x, obstacles):
    """2B LiDAR taraması üretir; çıktı robot çerçevesindedir."""
    beam_angles = np.linspace(-math.pi, math.pi, LIDAR_NUM_BEAMS, endpoint=False)
    points = []

    for local_angle in beam_angles:
        global_angle = x[2, 0] + local_angle
        beam_range = MAX_LIDAR_RANGE

        for obstacle in obstacles:
            hit = ray_circle_intersection(x, global_angle, obstacle)
            if hit is not None and hit < beam_range:
                beam_range = hit

        if beam_range <= LIDAR_THRESHOLD:
            px = beam_range * math.cos(local_angle)
            py = beam_range * math.sin(local_angle)
            points.append([px, py, beam_range, local_angle])

    if not points:
        return np.empty((0, 4))

    return np.array(points)


def cluster_lidar_points(lidar_points):
    """Mesafe eşikleme sonrası komşu noktaları kümeleyerek engel adayları üretir."""
    if lidar_points.size == 0:
        return []

    sorted_points = lidar_points[np.argsort(lidar_points[:, 3])]
    clusters = []
    current_cluster = [sorted_points[0]]

    for point in sorted_points[1:]:
        previous = current_cluster[-1]
        distance = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if distance <= CLUSTER_DISTANCE_THRESHOLD:
            current_cluster.append(point)
        else:
            clusters.append(np.array(current_cluster))
            current_cluster = [point]

    clusters.append(np.array(current_cluster))

    detected_obstacles = []
    for cluster in clusters:
        centroid_x = float(np.mean(cluster[:, 0]))
        centroid_y = float(np.mean(cluster[:, 1]))
        spreads = np.sqrt((cluster[:, 0] - centroid_x) ** 2 + (cluster[:, 1] - centroid_y) ** 2)
        radius = float(max(np.max(spreads), 0.15) + 0.15)
        detected_obstacles.append((centroid_x, centroid_y, radius))

    return detected_obstacles


def lidar_clusters_to_world(x_est, detected_obstacles_robot):
    if not detected_obstacles_robot:
        return []

    world_obstacles = []
    for ox_r, oy_r, radius in detected_obstacles_robot:
        world_point = transform_robot_to_world(x_est[0, 0], x_est[1, 0], x_est[2, 0], np.array([[ox_r, oy_r]]))[0]
        world_obstacles.append((float(world_point[0]), float(world_point[1]), float(radius)))

    return world_obstacles


def simulate_encoder_measurement(u_true):
    # Tekerlek enkoderi girdisi, diferansiyel sürüş hızlarından türetilir.
    v_true = float(u_true[0, 0])
    w_true = float(u_true[1, 0])
    left_distance = (v_true - 0.5 * WHEEL_BASE * w_true) * DT
    right_distance = (v_true + 0.5 * WHEEL_BASE * w_true) * DT

    left_distance += np.random.randn() * ENCODER_V_NOISE_STD * DT
    right_distance += np.random.randn() * ENCODER_V_NOISE_STD * DT

    v_odom = (left_distance + right_distance) / (2.0 * DT)
    w_odom = (right_distance - left_distance) / (WHEEL_BASE * DT)
    return np.array([[v_odom], [w_odom]])


def simulate_imu_measurement(x_true):
    # IMU, robotun yönelimini ölçen ikinci sensör olarak kullanılır.
    yaw_true = float(x_true[2, 0])
    yaw_measurement = wrap_to_pi(yaw_true + np.random.randn() * IMU_YAW_NOISE_STD)
    return yaw_measurement

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
    """ IMU destekli açı ölçüm modeli """
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
    """Genişletilmiş Kalman Filtresi: enkoder tahmini + IMU düzeltmesi."""
    # Tahmin (Predict) - Dead Reckoning
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    # Güncelleme (Update) - IMU sensörü ile yönelim düzeltmesi
    H = np.array([[0, 0, 1]])
    zPred = np.array([[xPred[2, 0]]])
    y = np.array([[wrap_to_pi(z - zPred[0, 0])]])
    S = H @ PPred @ H.T + R_IMU
    K = PPred @ H.T @ np.linalg.inv(S)
    
    xEst = xPred + K @ y
    xEst[2, 0] = wrap_to_pi(xEst[2, 0])
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
    yaw_error = wrap_to_pi(yaw_error)
    
    v = math.hypot(fx, fy)
    v = np.clip(v, 0.0, 1.5)  # Maksimum hız sınırı
    w = 2.0 * yaw_error       # Basit oransal açısal hız kontrolcüsü
    w = np.clip(w, -1.5, 1.5) # Açısal hız sınırını biraz artırarak daha keskin manevra sağlandı
    
    return np.array([[v], [w]])


def compare_errors(x_true, x_dr, x_est):
    # Gerçek yol ile dead reckoning ve EKF arasındaki hata karşılaştırılır.
    dr_error = math.hypot(x_true[0, 0] - x_dr[0, 0], x_true[1, 0] - x_dr[1, 0])
    est_error = math.hypot(x_true[0, 0] - x_est[0, 0], x_true[1, 0] - x_est[1, 0])
    return dr_error, est_error

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
    xPlan = np.copy(xTrue)  # Planlanan yol: kontrol girdisinin gürültüsüz referansı
    PEst = np.eye(3)
    
    # Geçmiş verileri kaydetmek için (Grafikler için)
    hxTrue, hxDR, hxEst, hxPlan = [], [], [], []
    hErrorDR, hErrorEst, hYawIMU = [], [], []
    last_lidar_points_robot = np.empty((0, 4))
    last_detected_obstacles_robot = []
    last_detected_obstacles_world = []

    while time < SIM_TIME:
        time += DT

        # 2B LiDAR verisi üretimi, mesafe eşikleme ve engel kümeleme burada yapılır.
        lidar_points_robot = simulate_lidar_scan(xTrue, OBSTACLES)
        detected_obstacles_robot = cluster_lidar_points(lidar_points_robot)
        detected_obstacles_world = lidar_clusters_to_world(xEst, detected_obstacles_robot)

        if lidar_points_robot.size > 0:
            last_lidar_points_robot = lidar_points_robot
        if detected_obstacles_robot:
            last_detected_obstacles_robot = detected_obstacles_robot
        if detected_obstacles_world:
            last_detected_obstacles_world = detected_obstacles_world
        
        # Kontrol girdisini hesapla (APF)
        # Dinamik yeniden yönlendirme: LiDAR ile algılanan engeller APF'ye beslenir.
        active_obstacles = last_detected_obstacles_world if last_detected_obstacles_world else OBSTACLES
        u = potential_field_control(xEst, GOAL, active_obstacles)

        # Planlanan yol: aynı kontrol girdisinin gürültüsüz referans izi.
        xPlan = motion_model(xPlan, u)
        xPlan[2, 0] = wrap_to_pi(xPlan[2, 0])
        
        # Gerçek durumu simüle et (Süreç gürültüsü ile)
        u_true = u + np.random.randn(2, 1) * np.array([[0.05], [0.05]])
        xTrue = motion_model(xTrue, u_true)
        
        # Tekerlek enkoderi dead reckoning için kullanılır.
        u_odom = simulate_encoder_measurement(u_true)
        xDR = motion_model(xDR, u_odom)
        xDR[2, 0] = wrap_to_pi(xDR[2, 0])
        
        # IMU, EKF güncellemesinde ikinci sensör olarak kullanılır.
        imu_yaw = simulate_imu_measurement(xTrue)
        
        # EKF ile sensör füzyonlu konum tahmini yapılır.
        xEst, PEst = ekf_estimation(xEst, PEst, imu_yaw, u_odom)
        
        # Gerçek yol ile tahmin edilen yol arasındaki hata karşılaştırılır.
        dr_error, est_error = compare_errors(xTrue, xDR, xEst)
        
        # Geçmişi kaydet
        hxTrue.append(xTrue[:, 0].tolist())
        hxDR.append(xDR[:, 0].tolist())
        hxEst.append(xEst[:, 0].tolist())
        hxPlan.append(xPlan[:, 0].tolist())
        hErrorDR.append(dr_error)
        hErrorEst.append(est_error)
        hYawIMU.append(imu_yaw)
        
        # Hedefe varış kontrolü
        if math.hypot(xTrue[0, 0] - GOAL[0], xTrue[1, 0] - GOAL[1]) < 0.5:
            print("Hedefe ulaşıldı!")
            break

    hxTrue = np.array(hxTrue)
    hxDR = np.array(hxDR)
    hxEst = np.array(hxEst)
    hxPlan = np.array(hxPlan)
    
    # ==========================================
    # 5. GÖRSELLEŞTİRME VE GRAFİKLER
    # ==========================================
    # Grafikleri ekranda göster
    time_arr = np.arange(0, len(hxTrue) * DT, DT) if len(hxTrue) > 0 else np.array([])

    # Sayfa 1: Ortam ve yollar (Planlanan, Gerçek, EKF, DR)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(START[0], START[1], "go", markersize=10, label="Başlangıç")
    plt.plot(GOAL[0], GOAL[1], "r*", markersize=10, label="Hedef")
    for (ox, oy, orad) in OBSTACLES:
        circle = plt.Circle((ox, oy), orad, color='gray')
        plt.gca().add_patch(circle)
    if len(hxPlan) > 0:
        plt.plot(hxPlan[:, 0], hxPlan[:, 1], "--", color="tab:orange", label="Planlanan Yol")
    if len(hxTrue) > 0:
        plt.plot(hxTrue[:, 0], hxTrue[:, 1], "-b", label="Gerçek Yol")
    if len(hxDR) > 0:
        plt.plot(hxDR[:, 0], hxDR[:, 1], "-k", label="Dead Reckoning")
    if len(hxEst) > 0:
        plt.plot(hxEst[:, 0], hxEst[:, 1], "-r", label="EKF Tahmini")
    plt.title("Ortam ve Yol Haritası")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    plt.close(fig)

    # Sayfa 2: LiDAR ham verisi ve filtrelenmiş küme merkezleri
    fig = plt.figure(figsize=(8, 6))
    if last_lidar_points_robot.size > 0:
        plt.scatter(last_lidar_points_robot[:, 0], last_lidar_points_robot[:, 1], s=18, c=last_lidar_points_robot[:, 2], cmap='viridis', label='Ham LiDAR Noktaları')
    for (ox, oy, orad) in last_detected_obstacles_robot:
        circle = plt.Circle((ox, oy), orad, fill=False, color='orange', linestyle='--')
        plt.gca().add_patch(circle)
        plt.scatter(ox, oy, s=80, marker='x', color='red', label='Filtrelenmiş Küme Merkezi')
    plt.title("LiDAR: Ham Noktalar ve Filtrelenmiş Küme Merkezleri")
    plt.xlabel("Robot Çerçevesi X (m)")
    plt.ylabel("Robot Çerçevesi Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    plt.close(fig)

    # Sayfa 3: x(t)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(time_arr, hxTrue[:, 0], "-b", label="Gerçek x(t)")
    plt.plot(time_arr, hxEst[:, 0], "-r", label="EKF x(t)")
    plt.plot(time_arr, hxDR[:, 0], "-k", alpha=0.65, label="Dead Reckoning x(t)")
    plt.title("Lokalizasyon - x(t)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("x (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close(fig)

    # Sayfa 4: y(t)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(time_arr, hxTrue[:, 1], "-b", label="Gerçek y(t)")
    plt.plot(time_arr, hxEst[:, 1], "-r", label="EKF y(t)")
    plt.plot(time_arr, hxDR[:, 1], "-k", alpha=0.65, label="Dead Reckoning y(t)")
    plt.title("Lokalizasyon - y(t)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close(fig)

    # Sayfa 5: theta(t)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(time_arr, hxTrue[:, 2], "-b", label="Gerçek θ(t)")
    plt.plot(time_arr, hxEst[:, 2], "-r", label="EKF θ(t)")
    plt.plot(time_arr, hxDR[:, 2], "-k", alpha=0.65, label="Dead Reckoning θ(t)")
    plt.title("Lokalizasyon - θ(t)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Açı (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close(fig)

    # Sayfa 6: Hata analizi
    fig = plt.figure(figsize=(8, 4))
    plt.plot(time_arr, hErrorDR, "-k", label="Dead Reckoning Hatası")
    plt.plot(time_arr, hErrorEst, "-g", label="EKF Hatası")
    rmse_dr = np.sqrt(np.mean(np.array(hErrorDR) ** 2)) if len(hErrorDR) > 0 else 0.0
    rmse_est = np.sqrt(np.mean(np.array(hErrorEst) ** 2)) if len(hErrorEst) > 0 else 0.0
    plt.title(f"Hata Analizi | DR RMSE: {rmse_dr:.3f} m | EKF RMSE: {rmse_est:.3f} m")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Konum Hatası (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close(fig)

    print('Grafikler ekranda gösterildi.')

    print("\n--- Teslim Durumu Özeti ---")
    print("- LiDAR 2B tarama: var")
    print("- Mesafe eşikleme ve kümelenme: var")
    print("- Tekerlek enkoderi: var")
    print("- IMU: var")
    print("- Kalman filtresi: EKF var")
    print("- Dead reckoning karşılaştırması: var")
    print("- Engel kaçınma ve reaktif davranış: var")
    print("- Dinamik yeniden planlama: kısmi, APF tabanlı yerel yeniden yönlenme")
    
    # Oluşturulan engellerin konumlarını yazdır
    print("\n" + "="*50)
    print("Rastgele Oluşturulan Engel Konumları:")
    print("="*50)
    for i, (ox, oy, orad) in enumerate(OBSTACLES, 1):
        print(f"Engel {i}: Merkez=({ox:.2f}, {oy:.2f}), Yarıçap={orad:.2f}")

if __name__ == '__main__':
    main()
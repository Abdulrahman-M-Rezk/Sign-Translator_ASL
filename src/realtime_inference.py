import os
# خفي رسائل التنسرفلو المزعجة
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import mediapipe as mp   
import cv2
import numpy as np
import json
import time

# ================= CONFIG (عدل هنا) =================
# 1. مسار الموديل اللي دربته (تأكد من الاسم والمكان)
MODEL_PATH = '/mnt/Hub_1/Mix/Projects/Graduation -Project/Notebooks/best_bilstm_local.keras' # أو الامتداد .h5 حسب ما حفظته

# 2. مسار ملف القاموس
LABEL_MAP_PATH = r'/mnt/Hub_1/Mix/Projects/Graduation -Project/Data/label_map.json'

# إعدادات ثابتة (زي التدريب بالظبط)
SEQUENCE_LENGTH = 50
CONF_THRESH = 0.5

# ================= Load Resources =================
print("⏳ Loading Model & Resources...")

# تحميل الموديل
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# تحميل القاموس (وعكسه عشان نحول الرقم لاسم)
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
# عكس القاموس: {0: 'apple', 1: 'book', ...}
inv_label_map = {v: k for k, v in label_map.items()}
print(f"✅ Label Map Loaded ({len(label_map)} classes).")

# ================= Helper Functions (Preprocessing) =================
# نفس الدوال اللي استخدمناها في التدريب بالظبط (مهم جداً تكون متطابقة)
mp_holistic = mp.solutions.holistic

def normalize_hand(pts):
    ref = pts[0].copy()
    scale = np.linalg.norm(pts[9] - ref)
    if scale < 1e-6: scale = 1.0
    return (pts - ref) / scale

def choose_best_hands(multi_hand_landmarks, multi_handedness):
    chosen = {}
    if multi_hand_landmarks is None or multi_handedness is None:
        return chosen
    for lm, hd in zip(multi_hand_landmarks, multi_handedness):
        label = hd.classification[0].label.upper()
        conf  = float(hd.classification[0].score)
        if conf < CONF_THRESH: continue
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        chosen[label] = pts
    return chosen

def compute_torso_center_and_scale(pose_landmarks):
    torso_center = np.array([0.5, 0.5], dtype=np.float32)
    torso_scale = 1.0
    try:
        ps = pose_landmarks
        def get_xy(idx):
            lm = ps.landmark[idx]
            return np.array([lm.x, lm.y], dtype=np.float32)
        
        left_sh, right_sh = get_xy(11), get_xy(12)
        left_hip, right_hip = get_xy(23), get_xy(24)
        
        shoulder_center = (left_sh + right_sh) / 2.0
        hip_center = (left_hip + right_hip) / 2.0
        torso_center = (shoulder_center + hip_center) / 2.0
        
        shoulder_dist = np.linalg.norm(left_sh - right_sh)
        hip_dist = np.linalg.norm(left_hip - right_hip)
        torso_scale = max(shoulder_dist, hip_dist, 1e-6)
    except: pass
    return torso_center, float(torso_scale)

def extract_features(results):
    # 198 Feature Vector Extraction
    feat = np.zeros(198, dtype=np.float32)
    
    torso_center = np.array([0.5, 0.5], dtype=np.float32)
    torso_scale = 1.0

    if results.pose_landmarks:
        torso_center, torso_scale = compute_torso_center_and_scale(results.pose_landmarks)
        pose_xy = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        pose_norm = (pose_xy - torso_center[None, :]) / torso_scale
        feat[0:66] = pose_norm.flatten()

    chosen = choose_best_hands(getattr(results, 'multi_hand_landmarks', None),
                               getattr(results, 'multi_handedness', None))

    # Left Hand
    if 'LEFT' in chosen:
        left_pts = chosen['LEFT']
        feat[66:129] = normalize_hand(left_pts)[:, :3].flatten()
        wrist = left_pts[0]
        wrist_rel = np.array([(wrist[0] - torso_center[0]) / torso_scale,
                              (wrist[1] - torso_center[1]) / torso_scale,
                              wrist[2] / max(torso_scale, 1e-6)], dtype=np.float32)
        feat[129:132] = wrist_rel

    # Right Hand
    if 'RIGHT' in chosen:
        right_pts = chosen['RIGHT']
        feat[132:195] = normalize_hand(right_pts)[:, :3].flatten()
        wrist = right_pts[0]
        wrist_rel = np.array([(wrist[0] - torso_center[0]) / torso_scale,
                              (wrist[1] - torso_center[1]) / torso_scale,
                              wrist[2] / max(torso_scale, 1e-6)], dtype=np.float32)
        feat[195:198] = wrist_rel
        
    return feat

# ================= Main Loop =================
print("🎥 Opening Camera...")
cap = cv2.VideoCapture(0) # 0 للكاميرا الأساسية، 1 لو عندك كاميرا تانية

# مخزن للفريمات (Sliding Window)
sequence = [] 
current_word = "Waiting..."
probability = 0.0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. تجهيز الصورة
        image = cv2.flip(frame, 1) # مراية
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. استخراج النقاط
        results = holistic.process(image_rgb)
        
        # 3. رسم النقاط على الشاشة (للتوضيح)
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 4. استخراج الميزات الرقمية
        keypoints = extract_features(results)
        sequence.append(keypoints)
        
        # نحتفظ بآخر 50 فريم بس
        sequence = sequence[-SEQUENCE_LENGTH:]

        # 5. التوقع (لما نجمع 50 فريم)
        if len(sequence) == SEQUENCE_LENGTH:
            # تحويل لشكل يقبله الموديل (1, 50, 198)
            input_data = np.expand_dims(sequence, axis=0)
            
            # التنبؤ
            res = model.predict(input_data, verbose=0)[0]
            
            # ناخد أعلى احتمال
            best_idx = np.argmax(res)
            probability = res[best_idx]
            
            # لو الاحتمال عالي كفاية، نعرض الكلمة
            if probability > 0.6: # ممكن تغير الـ Threshold ده
                current_word = inv_label_map[best_idx]
            else:
                current_word = "..."

        # 6. الكتابة على الشاشة
        # مستطيل خلفية للكلام
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        # الكلمة
        cv2.putText(image, f"{current_word} ({probability*100:.1f}%)", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # شريط التقدم (بيوريك جمعت كام فريم من الـ 50)
        progress = len(sequence) / SEQUENCE_LENGTH
        cv2.rectangle(image, (0, 40), (int(640 * progress), 45), (0, 255, 0), -1)

        cv2.imshow('Sign Language Translator (BiLSTM)', image)

        # الخروج بزر 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
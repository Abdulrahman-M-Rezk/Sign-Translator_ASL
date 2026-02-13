import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import time

# --- إعدادات الملفات ---
MODEL_PATH = "best_htcn_model.keras"  # تأكد أن اسم ملف الموديل صحيح
MAP_PATH = "Models/best_htcn_model.keras/sign_to_prediction_index_map.json"

# --- إعدادات الموديل ---
SEQ_LEN = 50       # عدد الفريمات (Buffer)
THRESHOLD = 0.6    # أقل نسبة ثقة لقبول الكلمة (60%)

# --- تحميل القاموس (Mapping) ---
try:
    with open(MAP_PATH, 'r') as f:
        data = json.load(f)
        # نحتاج عكس القاموس: المفتاح يكون الرقم والقيمة تكون الكلمة
        # {0: "TV", 1: "after", ...}
        idx_to_word = {v: k for k, v in data.items()}
    print(f"✅ Loaded {len(idx_to_word)} classes from JSON.")
except Exception as e:
    print(f"❌ Error loading JSON: {e}")
    exit()

# --- تحميل الموديل ---
try:
    print("🔄 Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading Model. Make sure tensorflow is installed properly.\n{e}")
    exit()

# --- دوال المعالجة (Normalization Logic) ---
# نفس المنطق المستخدم في التدريب لضمان الدقة
def get_hand_features(hand_landmarks, body_center):
    if hand_landmarks:
        hand_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = hand_np[0, :]
        local_hand = hand_np - wrist  # Hand relative to wrist
        wrist_context = wrist - body_center # Wrist relative to body
        return np.concatenate([local_hand.flatten(), wrist_context])
    else:
        return np.zeros(66) # (21*3 + 3)

def get_pose_features(pose_landmarks):
    if pose_landmarks:
        pose_np = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])
        # حساب مركز الجسم (منتصف الكتفين)
        left_shoulder = pose_np[11, :]
        right_shoulder = pose_np[12, :]
        body_center = (left_shoulder + right_shoulder) / 2.0
        
        pose_centered = pose_np - body_center
        pose_xy = pose_centered[:, :2] # نأخذ x, y فقط للجسم
        return pose_xy.flatten(), body_center
    else:
        return np.zeros(66), np.zeros(3)

# --- التشغيل الرئيسي ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sequence = [] # لتخزين الـ 50 فريم
current_word = "Waiting..."

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # معالجة ميديا بايب
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # رسم النقاط (للتأكد أن الكاميرا تراك)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- استخراج البيانات ---
        pose_vec, body_center = get_pose_features(results.pose_landmarks)
        lh_vec = get_hand_features(results.left_hand_landmarks, body_center)
        rh_vec = get_hand_features(results.right_hand_landmarks, body_center)

        # دمج البيانات (فريم واحد)
        # الترتيب: Pose -> Left -> Right
        frame_features = np.concatenate([pose_vec, lh_vec, rh_vec])
        
        # إضافته للذاكرة
        sequence.append(frame_features)
        sequence = sequence[-SEQ_LEN:] # الاحتفاظ بآخر 50 فقط

        # --- التوقع (Inference) ---
        if len(sequence) == SEQ_LEN:
            # تجهيز الداتا (Batch Dimension)
            input_data = np.expand_dims(sequence, axis=0)
            
            # تشغيل الموديل
            start_t = time.time()
            res = model.predict(input_data, verbose=0)[0]
            latency = (time.time() - start_t) * 1000
            
            predicted_idx = np.argmax(res)
            confidence = res[predicted_idx]

            if confidence > THRESHOLD:
                word = idx_to_word[predicted_idx]
                current_word = f"{word} ({confidence:.0%})"
                color = (0, 255, 0) # أخضر
            else:
                current_word = "..."
                color = (0, 0, 255) # أحمر

            # عرض المعلومات على الشاشة
            cv2.putText(image, current_word, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Lat: {latency:.1f}ms", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Sign Language Project - Real Time', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
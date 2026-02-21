import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time
from collections import Counter

# ================= مسارات الملفات =================
MODEL_PATH = "/mnt/Hub_1/Mix/Projects/Graduation-Project/models/Kaggle_test/model.tflite"
LABEL_MAP_PATH = "/mnt/Hub_1/Mix/Projects/Graduation-Project/models/Kaggle_test/sign_to_prediction_index_map.json"

# ================= فئة بناء الجمل والتنعيم (Logic Layer) =================
class PredictionSystem:
    def __init__(self, stabilization_frames=5, silence_threshold=2.0):
        self.history = []
        self.stabilization_frames = stabilization_frames
        self.sentence_buffer = [] 
        self.last_word_time = time.time()
        self.silence_threshold = silence_threshold
        self.current_stable_word = None
        
    def add_prediction(self, word, confidence):
        if confidence > 0.4:
            self.history.append(word)
            self.history = self.history[-self.stabilization_frames:]
        else:
            # إذا كانت الثقة ضعيفة، نعتبرها إشارة فارغة (Noise)
            self.history.append("")
            self.history = self.history[-self.stabilization_frames:]

        if len(self.history) == self.stabilization_frames:
            most_common = Counter(self.history).most_common(1)[0]
            candidate_word, count = most_common
            
            if count >= 3 and candidate_word != "" and candidate_word != self.current_stable_word:
                self.current_stable_word = candidate_word
                self.sentence_buffer.append(candidate_word)
                self.last_word_time = time.time()
                return candidate_word
                
        return None

    def check_sentence_completion(self):
        if time.time() - self.last_word_time > self.silence_threshold:
            if self.sentence_buffer:
                buffer_copy = self.sentence_buffer.copy()
                self.sentence_buffer = [] 
                self.history = []
                self.current_stable_word = None
                return buffer_copy
        return None

# ================= الدوال المساعدة =================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# ================= تحميل الموديل =================
print("⏳ Loading resources...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    FIXED_FRAMES = 30
    interpreter.resize_tensor_input(input_index, [1, FIXED_FRAMES, 543, 3]) 
    interpreter.allocate_tensors()
    print("✅ Model Loaded & Memory Allocated.")
    
except Exception as e:
    try: 
        print("⚠️ Retrying allocation without batch dim...")
        interpreter.resize_tensor_input(input_index, [FIXED_FRAMES, 543, 3])
        interpreter.allocate_tensors()
        print("✅ Model Loaded (No Batch Dim).")
    except Exception as e2:
        print(f"❌ Error loading model: {e2}")
        exit()

try:
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
        idx_to_sign = {v: k for k, v in label_map.items()}
except:
    idx_to_sign = None

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    def to_array(landmarks, count):
        if landmarks:
            return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        return [[float('nan'), float('nan'), float('nan')]] * count

    face = to_array(results.face_landmarks, 468)
    lh = to_array(results.left_hand_landmarks, 21)
    pose = to_array(results.pose_landmarks, 33)
    rh = to_array(results.right_hand_landmarks, 21)
    return np.concatenate([face, lh, pose, rh])

# ================= الحلقة الرئيسية =================
cap = cv2.VideoCapture(0)
sequence = []
last_prediction = "Waiting..."
prediction_conf = 0.0
completed_sentence_display = ""

# تهيئة نظام بناء الجمل
engine = PredictionSystem(stabilization_frames=5, silence_threshold=2.0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ⏱️ 1. بدء حساب زمن المعالجة (Pipeline Latency)
        start_time = time.time()

        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-FIXED_FRAMES:]

        if len(sequence) == FIXED_FRAMES:
            try:
                input_data = np.array(sequence, dtype=np.float32)
                
                if len(interpreter.get_input_details()[0]['shape']) == 4:
                     input_data = np.expand_dims(input_data, axis=0)

                interpreter.set_tensor(input_index, input_data)
                interpreter.invoke()
                raw_output = interpreter.get_tensor(output_index)
                
                if raw_output.ndim == 2:
                    prediction_logits = raw_output[0]
                else:
                    prediction_logits = raw_output

                probs = softmax(prediction_logits)
                top_idx = np.argmax(probs)
                current_conf = probs[top_idx]
                
                word = idx_to_sign[top_idx] if idx_to_sign else str(top_idx)
                
                # --- دمج منطق التنعيم وبناء الجمل ---
                new_word = engine.add_prediction(word, current_conf)
                if new_word:
                    last_prediction = new_word
                    prediction_conf = current_conf

            except Exception as e:
                print(f"Runtime Error: {e}")
                sequence = [] 

        # فحص اكتمال الجملة
        completed_sentence = engine.check_sentence_completion()
        if completed_sentence:
            completed_sentence_display = " ".join(completed_sentence)
            print(f"🚀 Ready for LLM: {completed_sentence_display}")

        # ⏱️ 2. إيقاف المؤقت (نهاية المعالجة الحسابية)
        end_time = time.time()
        pipeline_latency_ms = (end_time - start_time) * 1000

        # ================= العرض على الشاشة =================
        cv2.rectangle(image, (0,0), (640, 100), (0,0,0), -1)
        
        # عرض الكلمة الحالية
        cv2.putText(image, f"Word: {last_prediction} ({prediction_conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # عرض الجملة المكتملة
        cv2.putText(image, f"Sentence: {completed_sentence_display}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # عرض زمن الوصول بلون ديناميكي لضمان قيد المشروع (200ms)
        latency_color = (0, 255, 0) if pipeline_latency_ms <= 200 else (0, 0, 255)
        cv2.putText(image, f"Latency: {pipeline_latency_ms:.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, latency_color, 2)

        cv2.imshow('SignSense Pro - Logic Integrated', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
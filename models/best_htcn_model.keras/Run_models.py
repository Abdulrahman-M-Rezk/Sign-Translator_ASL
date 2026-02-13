import tensorflow as tf
import numpy as np

# 1. تحميل الموديل (غير 'model_name.keras' لاسم الملف اللي حملته)
model_path = 'model.weights.h5' 
model = tf.keras.models.load_model(model_path)

print("تم تحميل الموديل بنجاح! ✅")

# 2. عرض مواصفات الموديل (عشان نعرف محتاج مدخلات إيه)
model.summary()
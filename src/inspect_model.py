import tensorflow as tf
import numpy as np

# مسار الموديل
MODEL_PATH = "/mnt/Hub_1/Mix/Projects/Graduation-Project/models/Kaggle_test//model.tflite"

print(f"🕵️‍♂️ Inspecting Model: {MODEL_PATH} ...\n")

try:
    # 1. تحميل المفسر
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # 2. فحص التوقيعات (Signatures) - موديلات المسابقات تعتمد عليها
    signatures = interpreter.get_signature_list()
    print(f"✅ Signatures Found: {signatures}")
    
    if not signatures:
        print("⚠️ No signatures found. Using default input/output details.")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n--- Input Details ---")
        for i, detail in enumerate(input_details):
            print(f"[Input {i}] Name: {detail['name']}")
            print(f"          Shape: {detail['shape']}")
            print(f"          Type:  {detail['dtype']}")
            print(f"          Index: {detail['index']}")

        print("\n--- Output Details ---")
        for i, detail in enumerate(output_details):
            print(f"[Output {i}] Name: {detail['name']}")
            print(f"           Shape: {detail['shape']}")
            print(f"           Type:  {detail['dtype']}")
            print(f"           Index: {detail['index']}")
            
    else:
        # فحص كل توقيع (غالباً يهمنا توقيع اسمه 'serving_default')
        for key in signatures.keys():
            print(f"\n🔑 Analyzing Signature: '{key}'")
            runner = interpreter.get_signature_runner(key)
            
            # تفاصيل المدخلات
            print("   --- Inputs ---")
            inputs = runner.get_input_details()
            for name, detail in inputs.items():
                print(f"   Name: '{name}'")
                print(f"   Shape: {detail['shape']}") # ركز هنا جداً
                print(f"   Type:  {detail['dtype']}")
            
            # تفاصيل المخرجات
            print("   --- Outputs ---")
            outputs = runner.get_output_details()
            for name, detail in outputs.items():
                print(f"   Name: '{name}'")
                print(f"   Shape: {detail['shape']}")
                print(f"   Type:  {detail['dtype']}")

except Exception as e:
    print(f"❌ Error: {e}")
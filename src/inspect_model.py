import argparse
import tensorflow as tf
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a TFLite model's inputs/outputs and signatures.")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/Kaggle_test/model.tflite",
        help="Path to the .tflite model file (default: models/Kaggle_test/model.tflite)",
    )
    return parser.parse_args()


def main(model_path: str) -> None:
    print(f"🕵️‍♂️ Inspecting Model: {model_path} ...\n")

    try:
        # 1. تحميل المفسر
        interpreter = tf.lite.Interpreter(model_path=model_path)
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
            
            for key in signatures.keys():
                print(f"\n🔑 Analyzing Signature: '{key}'")
                runner = interpreter.get_signature_runner(key)

                # تفاصيل المدخلات
                print("   --- Inputs ---")
                inputs = runner.get_input_details()
                for name, detail in inputs.items():
                    print(f"   Name: '{name}'")
                    print(f"   Shape: {detail['shape']}")  # ركز هنا جداً
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


if __name__ == "__main__":
    args = parse_args()
    main(args.model_path)
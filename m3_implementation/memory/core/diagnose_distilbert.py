# m3_implementation/memory/core/diagnose_distilbert.py
# Run this to find exactly why DistilBERT fails to load.
# Place in memory/core/ and run:
#   python -m memory.core.diagnose_distilbert

import sys, os, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("DISTILBERT LOAD DIAGNOSTIC")
print("="*60)

# Step 1: Check .env value
model_path_env = os.getenv("DISTILBERT_MODEL_PATH")
print(f"\n1. DISTILBERT_MODEL_PATH in .env:")
print(f"   Raw value: {repr(model_path_env)}")

if not model_path_env:
    print("   ERROR: Value is None or empty!")
    print("   Fix: Make sure your .env file contains:")
    print("   DISTILBERT_MODEL_PATH=m3_implementation/adaptive_rag/distilbert_training/outputs/best_model")
    sys.exit(1)

# Step 2: Resolve the absolute path
this_file  = os.path.abspath(__file__)
print(f"\n2. This file location:")
print(f"   {this_file}")

# pipeline.py uses 3 levels up from memory/core/
core_dir        = os.path.dirname(this_file)          # memory/core/
memory_dir      = os.path.dirname(core_dir)           # memory/
m3_dir          = os.path.dirname(memory_dir)         # m3_implementation/
project_root    = os.path.dirname(m3_dir)             # project root

print(f"\n3. Computed project root:")
print(f"   {project_root}")
print(f"   Exists: {os.path.exists(project_root)}")

if os.path.isabs(model_path_env):
    model_path = model_path_env
else:
    model_path = os.path.normpath(os.path.join(project_root, model_path_env))

print(f"\n4. Resolved model path:")
print(f"   {model_path}")
print(f"   Exists: {os.path.exists(model_path)}")

if not os.path.exists(model_path):
    print("\n   ERROR: Path does not exist!")
    print("   Listing what IS in the distilbert_training folder...")
    dt_dir = os.path.normpath(os.path.join(
        project_root, "m3_implementation", "adaptive_rag", "distilbert_training"
    ))
    if os.path.exists(dt_dir):
        for item in os.listdir(dt_dir):
            print(f"     {item}")
        outputs_dir = os.path.join(dt_dir, "outputs")
        if os.path.exists(outputs_dir):
            print(f"   Contents of outputs/:")
            for item in os.listdir(outputs_dir):
                print(f"     {item}")
    sys.exit(1)

# Step 3: List model files
print(f"\n5. Files inside best_model/:")
for f in os.listdir(model_path):
    size = os.path.getsize(os.path.join(model_path, f))
    print(f"   {f}  ({size:,} bytes)")

# Step 4: Try adding distilbert_training to sys.path and importing predict
distilbert_training_dir = os.path.normpath(
    os.path.join(model_path, '..', '..')
)
print(f"\n6. distilbert_training dir:")
print(f"   {distilbert_training_dir}")
print(f"   Exists: {os.path.exists(distilbert_training_dir)}")

predict_py = os.path.join(distilbert_training_dir, "predict.py")
config_py  = os.path.join(distilbert_training_dir, "config.py")
print(f"   predict.py exists: {os.path.exists(predict_py)}")
print(f"   config.py exists:  {os.path.exists(config_py)}")

if distilbert_training_dir not in sys.path:
    sys.path.insert(0, distilbert_training_dir)
print(f"\n7. sys.path[0]: {sys.path[0]}")

# Step 5: Try importing and loading
print(f"\n8. Attempting to import Predictor...")
try:
    from predict import Predictor
    print("   predict.py imported OK")
    
    print(f"\n9. Attempting to load model from:")
    print(f"   {model_path}")
    predictor = Predictor(model_dir=model_path)
    print("\n   SUCCESS! Model loaded.")
    
    # Quick test
    result = predictor.predict([], "I want a black dress")
    print(f"\n10. Quick prediction test:")
    print(f"    Label: {result['label_name']}")
    print(f"    Confidence: {result['confidence']:.1%}")
    print("\nDIAGNOSTIC COMPLETE — DistilBERT is working correctly.")
    
except Exception as e:
    print(f"   FAILED: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

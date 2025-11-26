import subprocess
import sys
import time
import os

# List of scripts to run in order
# Each tuple is (Description, Script Filename)
STEPS = [
    ("Preprocessing", "data_preprocessing.py"),
    ("ConvNeXt Training", "convnext_model.py"),
    ("DenseNet121 Training", "densenet121_model.py"),
    ("ResNet18 Training", "resnet18_model.py"),
    ("ResNet50 Training", "resnet50_model.py"),
    ("Swin Transformer Training", "swin_transformer_model.py"),
    ("Swin Multi-View Training", "swin_transformer_multiview_model.py"),
    ("Model Comparison", "compare_models.py")
]

def run_pipeline():
    print("CBIS-DDSM Automated Pipeline")
    print(f"Scheduled jobs: {len(STEPS)}")
    
    total_start = time.time()
    
    for description, script_name in STEPS:
        if not os.path.exists(script_name):
            print(f"Error: Could not find {script_name}")
            continue
            
        print(f"STEP: {description}")
        print(f"Script: {script_name}")
        
        step_start = time.time()
        try:
            # Run the script and wait for it to complete
            # sys.executable ensures we use the same python interpreter (env)
            subprocess.run([sys.executable, script_name], check=True)
            
            elapsed = time.time() - step_start
            print(f"\nFinished {description} in {elapsed/60:.1f} minutes\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError occurred in {description}!")
            print(f"Exit code: {e.returncode}")
            print("Pipeline stopped.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"Pipeline Completed Successfully!")
    print(f"Total pipeline execution time: {total_elapsed/60:.1f} minutes")

if __name__ == "__main__":
    run_pipeline()

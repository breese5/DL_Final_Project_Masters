#!/usr/bin/env python
"""
setup.py

Runs the full project setup pipeline for the Masters Tournament Prediction System.
This includes data preprocessing, training the Naive, Traditional, and Deep Learning models,
embedding generation, and evaluation.

Author: Bryant Reese
"""

import subprocess
import os

def run_script(script_name):
    """Utility to run a script with subprocess and print status."""
    print(f"\n🚀 Running: {script_name}")
    result = subprocess.run(["python", f"scripts/{script_name}"], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {script_name} completed successfully.")
    else:
        print(f"❌ {script_name} failed.")
        print(result.stderr)

def main():
    print("🔧 Starting Full Project Setup...\n")

    # Sequence of setup scripts
    scripts_to_run = [
        "traditional_preprocessing.py",
        "naive_model.py",
        "traditional_model.py",
        "embed_bios_and_metadata.py",
        "deep_learning_model.py",
        "evaluate_models.py"
    ]

    for script in scripts_to_run:
        run_script(script)

    print("\n🏁 Setup Complete! All models trained and evaluated.")

if __name__ == "__main__":
    main()

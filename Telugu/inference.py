"""Synthesise the test sentences + variance-adaptor plots for Telugu.
   python Telugu/inference.py --restore_step 40000 [--speaker spk000] [--controls]"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))
os.chdir(ROOT)
from inference_core import main  # noqa: E402

if __name__ == "__main__":
    main("Telugu")

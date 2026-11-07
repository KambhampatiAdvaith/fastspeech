"""Train the Telugu FastSpeech 2 model.   python Telugu/train.py [--restore_step N]"""
import argparse
import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))
os.chdir(ROOT)
from utils.config import load_config  # noqa: E402

# load the reference training loop under another name (this file is also called train.py)
spec = importlib.util.spec_from_file_location("fs2_train", os.path.join(ROOT, "fastspeech2", "train.py"))
fs2_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs2_train)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Telugu/config.yaml")
    ap.add_argument("--restore_step", type=int, default=0)
    args = ap.parse_args()
    fs2_train.main(args, load_config(args.config))

"""Preprocessing for English.
  python English/preprocess.py select [--hf_token ...|--corpus DIR]   # build the 5h/1h subset  (raw_data/English)
  bash   scripts/run_mfa.sh English                                    # forced alignment
  python English/preprocess.py features                                # mel / pitch / energy / duration + split
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "fastspeech2"), os.path.join(ROOT, "scripts")]
os.chdir(ROOT)
from utils.config import load_config  # noqa: E402

LANG = "English"
CONFIG = f"{LANG}/config.yaml"

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if step == "select":
        import select_svarah as sel
        sel.main(rest)
    elif step == "features":
        from utils.pipeline import build_features
        pre, _, _ = load_config(CONFIG)
        build_features(pre, os.path.join(pre["path"]["raw_path"], "val_ids.txt"))
    else:
        sys.exit(__doc__)

"""Experiment settings shared by the baseline and every pruned arm."""
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen3-8B"
N_VALUES = [2, 4, 9]         # layers removed, one arm per value
SEED = 0

# Calibration (SlimOrca)
N_CALIB = 1000     # ReplaceMe fits T on all of these
N_CUT = 256        # the first N_CUT select the cut
MAX_LEN = 512
BATCH_SIZE = 2

# Safety generation
ENABLE_THINKING = True
MAX_NEW = 1536
GEN_BATCH = 2      # greedy fp16 outputs depend on batch size, so keep fixed

# Utility evaluation (lm-eval, 0-shot)
TASKS = ["piqa", "winogrande", "openbookqa", "truthfulqa_mc2"]
LM_EVAL_BATCH = 2

OUT_DIR = Path(__file__).resolve().parent / "outputs"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"

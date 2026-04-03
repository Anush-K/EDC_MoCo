import os
import torch

# --------------------------------------------------
# Detect whether running inside Google Colab
# --------------------------------------------------
try:
    from IPython import get_ipython
    ipython = get_ipython()
    IN_COLAB = ipython is not None and "google.colab" in str(ipython)
except ImportError:
    IN_COLAB = False

ENV = "colab" if IN_COLAB else "local"

# --------------------------------------------------
# Device
# --------------------------------------------------
if ENV == "colab":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Base directory
# --------------------------------------------------
if ENV == "colab":
    BASE_DIR = "/content/UAD_EDC_DIFF"
else:
    BASE_DIR = "/home/cs24d0008/EDC_SSL"

# --------------------------------------------------
# Project directories
# --------------------------------------------------
CODE_DIR = os.path.join(BASE_DIR, "EDC-master")

# BUSI dataset must be organised as:
#   BUSI/
#     train/NORMAL/
#     test/NORMAL/
#     test/ABNORMAL/
DATASET_DIR = os.path.join(BASE_DIR, "BUSI")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

SAVED_MODELS_DIR = os.path.join(CODE_DIR, "saved_models")

# --------------------------------------------------
# Pretty print
# --------------------------------------------------
def print_config():
    print("===== BUSI CONFIGURATION =====")
    print(f"Environment:     {ENV}")
    print(f"Device:          {device}")
    print(f"Base directory:  {BASE_DIR}")
    print(f"Code directory:  {CODE_DIR}")
    print(f"Dataset root:    {DATASET_DIR}")
    print(f"Train folder:    {TRAIN_DIR}")
    print(f"Test folder:     {TEST_DIR}")
    print(f"Saved models:    {SAVED_MODELS_DIR}")
    print("================================")
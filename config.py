from pathlib import Path
import torch
import clip
from torchvision.transforms import transforms

# D:\anaconda\python.exe zhao.py 
# MPIIFaceGaze
# EyeDiap
# Gaze360
# ETH-XGaze
TRAIN_RUN_NAME = "ResNet-50_eth_train"
TEST_RUN_NAME = "ResNet-50_eth_test"
CNN_MODEL = "ResNet-50"
TRAIN_DATASET_NAME = "Gaze360"
MEAN=1.44
# TRAIN_DATASET_NAME = "EyeDiap"
# TRAIN_DATASET_NAME = "ETH-XGaze"
# TRAIN_DATASET_NAME = "MPIIFaceGaze"
# TEST_DATASET_NAME = "ETH-XGaze"
# TEST_DATASET_NAME = "MPIIFaceGaze"
# TEST_DATASET_NAME = "EyeDiap"
TEST_DATASET_NAME = "Gaze360"

IS_TRAIN = False
TEST_EPOCH = 30
TEST_CHECKPOINT = f"epoch_{TEST_EPOCH}.pth"
DATASETS_PATH = Path("datasets")
TRAIN_IMAGES_PATH = DATASETS_PATH / TRAIN_DATASET_NAME / "GazeHub" / "Image"
TEST_IMAGES_PATH = DATASETS_PATH / TEST_DATASET_NAME / "GazeHub" / "Image"
# TRAIN_IMAGES_PATH = DATASETS_PATH / TRAIN_DATASET_NAME / "GazeHub" / "Image"
# TEST_IMAGES_PATH = DATASETS_PATH / TEST_DATASET_NAME / "GazeHub" / "Image"
# ClusterLabel is for EyeDiap
TRAIN_LABELS_PATH = DATASETS_PATH / TRAIN_DATASET_NAME / "GazeHub" / "Label"
TEST_LABELS_PATH = DATASETS_PATH / TEST_DATASET_NAME / "GazeHub" / "Label"
# TRAIN_LABELS_PATH = DATASETS_PATH / TRAIN_DATASET_NAME / "GazeHub" / "Label"
# TEST_LABELS_PATH = DATASETS_PATH / TEST_DATASET_NAME / "GazeHub" / "Label"
OUT_PATH = Path("out")
CHECKPOINTS_PATH = OUT_PATH / "checkpoints"
LOGS_PATH = OUT_PATH / "logs"
RUNS_PATH = OUT_PATH / "runs"
SEED = 0
is_ablation = True
ABLA_CONFIG = {
    'use_feature_1': True,
    'use_feature_2': True,
    'use_feature_3': True,
    'use_feature_4': True,  
}
NUM_EPOCHS = 30
BATCH_SIZE = 64
TEST_STEP = 10
SAVE_STEP = 10
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01
BETAS=(0.9, 0.999)
ETA_MIN = 1e-7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CNN_PREPROCESS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


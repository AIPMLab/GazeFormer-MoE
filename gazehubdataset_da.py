from PIL import Image
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Callable, Optional
import cv2
from easydict import EasyDict as edict
import copy
from config import *
from utils import *

# dataset by GazeHub


def _parse_first_comma_separated_floats(label_tokens, candidates=None):
    """Find the first token in label_tokens that contains commas and can be
    parsed into a numeric numpy array. If candidates (an iterable of
    indices) is provided, try those indices first for deterministic behavior.

    Raises a ValueError with a helpful message if no suitable token is found.
    """
    # Try candidate indices first (if any)
    if candidates is not None:
        for i in candidates:
            if i < len(label_tokens):
                tok = label_tokens[i].strip()
                if "," in tok:
                    try:
                        return np.array(tok.split(",")).astype("float")
                    except Exception:
                        # fall through and continue searching
                        pass

    # Fallback: scan all tokens and return the first comma-containing
    # token that successfully parses to floats.
    for tok in label_tokens:
        tok = tok.strip()
        if "," not in tok:
            continue
        parts = tok.split(",")
        try:
            return np.array(parts).astype("float")
        except Exception:
            continue

    raise ValueError(
        "Could not find a comma-separated numeric token in label tokens: %r" % (
            label_tokens[:10]
        )
    )


class DatasetMPIIFaceGazeByGazeHub(Dataset):
    __transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    __target_transform = lambda label: torch.tensor(label)
    if TRAIN_DATASET_NAME == "Gaze360" and TEST_DATASET_NAME == "MPIIFaceGaze":
        __coefficients = np.array([-1, -1, 1])
    else:
        __coefficients = np.array([1, 1, 1])

    def __init__(
        self,
        images_path: Path,
        label_paths: list[Path],
        transform: Optional[Callable] = None,
        other_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels = [
            line.split(" ")
            for p in label_paths
            for line in p.read_text(encoding="utf-8").splitlines()[1:]
        ]
        self.labels = [
            edict(
                Face=label[0],
                # 变量名开头不能是数字
                _3DGaze=(
                    _parse_first_comma_separated_floats(label, candidates=[5])
                    * DatasetMPIIFaceGazeByGazeHub.__coefficients
                ),
            )
            for label in self.labels
        ]
        self.labels = self.labels[:]

        self.transform = (
            transform
            if transform is not None
            else DatasetMPIIFaceGazeByGazeHub.__transform
        )
        self.other_transform = (
            other_transform
            if other_transform is not None
            else DatasetMPIIFaceGazeByGazeHub.__transform
        )
        self.target_transform = (
            target_transform
            if target_transform is not None
            else DatasetMPIIFaceGazeByGazeHub.__target_transform
        )

    def __getitem__(self, idx):
        face_path = self.images_path / self.labels[idx].Face
        # cv2.imread can return None if the file is missing or unreadable.
        # Use str() to ensure a proper path string is passed, and check the
        # result so we can raise a clear error instead of hitting
        # cv2.cvtColor with an empty source.
        face_img = cv2.imread(str(face_path))
        if face_img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {face_path}")
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        other_face_img = copy.deepcopy(face_img)
        face_img = self.transform(face_img)
        other_face_img = self.other_transform(other_face_img)
        label = self.target_transform(self.labels[idx]._3DGaze)
        return edict(face=face_img, other_face=other_face_img), label

    def __len__(self):
        return len(self.labels)


class DatasetEyeDiapByGazeHub(Dataset):
    # __transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )
    __transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    __target_transform = lambda label: torch.tensor(label)
    if TRAIN_DATASET_NAME == "Gaze360" and TEST_DATASET_NAME == "EyeDiap":
        __coefficients = np.array([-1, -1, 1])
    else:
        __coefficients = np.array([1, 1, 1])

    def __init__(
        self,
        images_path: Path,
        label_paths: list[Path],
        transform: Optional[Callable] = None,
        other_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels = [
            line.split(" ")
            for p in label_paths
            for line in p.read_text(encoding="utf-8").splitlines()[1:]
        ]
        self.labels = [
            edict(
                Face=label[0],
                # 变量名开头不能是数字
                _3DGaze=(
                    _parse_first_comma_separated_floats(label, candidates=[4])
                    * DatasetEyeDiapByGazeHub.__coefficients
                ),
            )
            for label in self.labels
        ]

        self.transform = (
            transform if transform is not None else DatasetEyeDiapByGazeHub.__transform
        )
        self.other_transform = (
            other_transform
            if other_transform is not None
            else DatasetEyeDiapByGazeHub.__transform
        )
        self.target_transform = (
            target_transform
            if target_transform is not None
            else DatasetEyeDiapByGazeHub.__target_transform
        )

    def __getitem__(self, idx):
        face_path = self.images_path / self.labels[idx].Face
        face_img = cv2.imread(str(face_path))
        if face_img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {face_path}")
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        other_face_img = copy.deepcopy(face_img)
        face_img = self.transform(face_img)
        other_face_img = self.other_transform(other_face_img)

        label = self.target_transform(self.labels[idx]._3DGaze)

        return edict(face=face_img, other_face=other_face_img), label

    def __len__(self):
        return len(self.labels)


DatasetGaze360ByGazeHub = DatasetEyeDiapByGazeHub

# Custom dataset: use train.label (Face paths usually under train/Face) but load images from test/Face
# by rewriting the path prefix. This supports the user's requirement to pair train.label with test/Face images.
class DatasetGaze360TrainLabelTestFace(Dataset):
    __transform = DatasetEyeDiapByGazeHub._DatasetEyeDiapByGazeHub__transform  # reuse augmentation
    __target_transform = lambda label: torch.tensor(label)

    def __init__(
        self,
        images_path: Path,
        label_paths: list[Path],
        transform: Optional[Callable] = None,
        other_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        replace_prefix: tuple[str, str] = ("train/Face", "test/Face"),
    ):
        super().__init__()
        self.images_path = images_path
        self.labels = [
            line.split(" ")
            for p in label_paths
            for line in p.read_text(encoding="utf-8").splitlines()[1:]
        ]
        # columns: Face Left Right Origin 3DGaze 2DGaze
        self.replace_prefix = replace_prefix
        rp_from, rp_to = replace_prefix
        proc = []
        for label in self.labels:
            face_rel = label[0].replace("\\", "/")
            if face_rel.startswith(rp_from):
                face_rel_mapped = rp_to + face_rel[len(rp_from):]
            else:
                face_rel_mapped = face_rel
            proc.append(
                edict(
                    Face=face_rel_mapped,
                    _3DGaze=_parse_first_comma_separated_floats(label, candidates=[4]),
                )
            )
        self.labels = proc
        self.transform = transform if transform is not None else DatasetGaze360TrainLabelTestFace.__transform
        self.other_transform = (
            other_transform if other_transform is not None else DatasetGaze360TrainLabelTestFace.__transform
        )
        self.target_transform = (
            target_transform if target_transform is not None else DatasetGaze360TrainLabelTestFace.__target_transform
        )

    def __getitem__(self, idx):
        face_path = self.images_path / self.labels[idx].Face
        face_img = cv2.imread(str(face_path))
        if face_img is None:
            raise FileNotFoundError(f"Gaze360(mapped) image missing: {face_path}")
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        other_face_img = copy.deepcopy(face_img)
        face_img = self.transform(face_img)
        other_face_img = self.other_transform(other_face_img)
        label = self.target_transform(self.labels[idx]._3DGaze)
        return edict(face=face_img, other_face=other_face_img), label

    def __len__(self):
        return len(self.labels)


class DatasetETHXGazeByGazeHub(Dataset):
    __transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    __target_transform = lambda label: torch.tensor(label)

    def __init__(
        self,
        images_path: Path,
        label_paths: list[Path],
        transform: Optional[Callable] = None,
        other_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels = [
            line.split(" ")
            for p in label_paths
            for line in p.read_text(encoding="utf-8").splitlines()[1:]
        ]
        self.labels = [
            edict(
                Face=label[0],
                # 变量名开头不能是数字
                # Convert ETH pitch,yaw to 3D CCS vector.
                # ETH pitch/yaw convention produces vectors pointing opposite
                # sign to MPIIFaceGaze (z positive). Negate to match the
                # common convention (z forward -> negative values in MPII files).
                _3DGaze=(
                    -gaze_pitch_yaw_to_ccs(
                        _parse_first_comma_separated_floats(label, candidates=[1])
                    )
                ),
            )
            for label in self.labels
        ]

        self.transform = (
            transform if transform is not None else DatasetETHXGazeByGazeHub.__transform
        )
        self.other_transform = (
            other_transform
            if other_transform is not None
            else DatasetETHXGazeByGazeHub.__transform
        )
        self.target_transform = (
            target_transform
            if target_transform is not None
            else DatasetETHXGazeByGazeHub.__target_transform
        )

    def __getitem__(self, idx):
        face_path = self.images_path / self.labels[idx].Face
        face_img = cv2.imread(str(face_path))
        if face_img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {face_path}")
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        other_face_img = copy.deepcopy(face_img)
        face_img = self.transform(face_img)
        other_face_img = self.other_transform(other_face_img)

        label = self.target_transform(self.labels[idx]._3DGaze)

        return edict(face=face_img, other_face=other_face_img), label

    def __len__(self):
        return len(self.labels)

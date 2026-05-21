
is_ablation = False  # 消融实验标志，True时不保存checkpoint

import numpy as np
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用性: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch使用的CUDA版本: {torch.version.cuda}")
    print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from gazehubdataset_da import (
    DatasetMPIIFaceGazeByGazeHub,
    DatasetEyeDiapByGazeHub,
    DatasetGaze360ByGazeHub,
    DatasetETHXGazeByGazeHub,
    DatasetGaze360TrainLabelTestFace,
)
from config import *
import torch.optim as optim
from utils import leave_one_out, one
from models import GEWithCLIPModel_zhao as GEWithCLIPModel
import torch.nn as nn
from model_zhao_test import TransformerDeepSeek_gaze
import math
import os
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Function
import copy


class TransformerConfigBase:
    num_layers: int = 12
    embed_dim: int = 512
    inter_dim: int = 2048
    num_heads: int = 8
    # n_routed_experts: int = 8
    # n_activated_experts: int = 4
    n_routed_experts: int = 4
    n_activated_experts: int = 2
    n_shared_experts: int = 2
    moe_inter_dim: int = 1024


config = TransformerConfigBase()


class ZhaoDataset(Dataset):
    def __init__(self, images_path: Path,label_paths: list[Path], ds_name: str):
        super().__init__()
        # Chen-Xianrui
        if ds_name == "MPIIFaceGaze":
            DS = DatasetMPIIFaceGazeByGazeHub
        elif ds_name == "EyeDiap":
            DS = DatasetEyeDiapByGazeHub
        elif ds_name == "Gaze360":
            DS = DatasetGaze360ByGazeHub
        elif ds_name == "ETH-XGaze":
            DS = DatasetETHXGazeByGazeHub
        self.inner_dataset = DS(
            images_path,  # 这里用传入的 images_path
            label_paths,
            CLIP_PREPROCESS,
            CNN_PREPROCESS,
        )

    def __getitem__(self, idx):
        _input, label = self.inner_dataset[idx]
        return _input, label

    def __len__(self):
        return len(self.inner_dataset)


class TargetOnlyDataset(Dataset):
    """Wrap a dataset so that it returns dummy labels for target-domain training.
    This prevents accidental leakage of ground-truth labels during adversarial
    domain-adaptation training.
    """
    def __init__(self, inner_dataset):
        self.inner_dataset = inner_dataset

    def __getitem__(self, idx):
        _input, _ = self.inner_dataset[idx]
        # return a dummy 3-dim zero label (same shape as gaze label)
        dummy_label = torch.zeros(3, dtype=torch.float32)
        return _input, dummy_label

    def __len__(self):
        return len(self.inner_dataset)


def process_batch(batch, model):
    # 从 DataLoader 中取出的 batch，_input 为输入结构体，label 为 gaze 标签，[B, 3]
    _input, label = batch
    _input.face = _input.face.to(DEVICE)
    _input.other_face = _input.other_face.to(DEVICE)
    # label 始终保持 float32, 便于后续 loss 计算的数值稳定
    label = label.to(DEVICE).float()

    # 利用 hook 获取 encoder_i 输出的完整 token 序列（来自 VisionTransformer）
    hook_outputs = {}

    def vt_hook(module, inp, output):
        tokens = inp[0].permute(1, 0, 2)
        if tokens.dtype != torch.float32:
            tokens = tokens.float()
        hook_outputs["full_tokens"] = tokens  # expect shape [B, L, d_model]

    hook_handle = model.model.visual.transformer.register_forward_hook(vt_hook)
    img_feats = model.encoder_i(_input.face)  # 在 autocast 中可能是半精度
    orig_dtype = img_feats.dtype
    hook_handle.remove()  # 及时移除 hook
    if "full_tokens" in hook_outputs:
        full_tokens = hook_outputs["full_tokens"]
        if full_tokens.dim() == 3:
            # 不强制转换 float32，保持与 encoder 输出一致，节省显存
            token_img_patch = full_tokens[:, 1:, :]
        else:
            token_img_patch = None
            print("Captured tokens have unexpected dimensions.")
    else:
        token_img_patch = None

    feature_1, feature_2, aux = model.compute_conditioned_features(img_feats)
    feature_1 = feature_1.to(orig_dtype)
    feature_2 = feature_2.to(orig_dtype)

    # 获取 CNN 特征图并生成局部 token
    feature_map = model.main_model(_input.other_face)["features"]  # shape: [B, C, H, W]
    B, C, H, W = feature_map.shape
    tokens_feature3 = feature_map.view(B, C, H * W).transpose(1, 2)  # shape: [B, H*W, C]
    # 这里假设 tokens_feature3 的通道数 C 为512（可根据实际情况调整）

    # 按消融配置控制各特征，若为False则自动补零张量
    features = {}
    features["label"] = label
    # feature_1
    if ABLA_CONFIG.get('use_feature_1', True):
        features["feature_1"] = feature_1
    else:
        features["feature_1"] = torch.zeros_like(feature_1)
    # feature_2
    if ABLA_CONFIG.get('use_feature_2', True):
        features["feature_2"] = feature_2
    else:
        features["feature_2"] = torch.zeros_like(feature_2)
    # feature_3
    if ABLA_CONFIG.get('use_feature_3', True):
        features["feature_3"] = tokens_feature3
    else:
        features["feature_3"] = torch.zeros_like(tokens_feature3)
    # feature_4
    if ABLA_CONFIG.get('use_feature_4', True):
        features["token_img_patch"] = token_img_patch
    else:
        if token_img_patch is not None:
            features["token_img_patch"] = torch.zeros_like(token_img_patch)
        else:
            features["token_img_patch"] = None
    # expose img_norm and sim_label for downstream pseudo-label generation
    # img_norm: [B, dim], sim_label: [B, n_labels]
    features["img_norm"] = aux["img_norm"].to(orig_dtype)
    # 相似度保留 float32 以便后续阈值判断（若使用伪标签）
    features["sim_label"] = aux["sim_label"]  # float32
    return features

def debug_check_tensor(name, t):
    if isinstance(t, torch.Tensor):
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[DEBUG][NaNFound] {name}: contains NaN/Inf (min={t.min().item() if t.numel()>0 else 'NA'}, max={t.max().item() if t.numel()>0 else 'NA'})")
            return False
    return True


def kabsch_rotation(A: np.ndarray, B: np.ndarray):
    """Compute the best-fit rotation matrix R that maps A -> B using Kabsch algorithm.
    A and B are arrays of shape [N,3]. Returns R (3x3) and rotated A.
    """
    # center
    assert A.shape == B.shape and A.shape[1] == 3
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    A_centered = A - A_mean
    B_centered = B - B_mean
    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    A_rot = (R @ A_centered.T).T + B_mean
    return R, A_rot


def mean_angle_between_sets(A: np.ndarray, B: np.ndarray):
    """Compute mean angular error (degrees) between corresponding 3D unit vectors in A and B.
    A and B should be shape [N,3]."""
    # normalize
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    cos = np.sum(An * Bn, axis=1).clip(-1.0, 1.0)
    angles = np.arccos(cos) * 180.0 / np.pi
    return float(np.mean(angles)), angles


def axis_permutation_sweep(A: np.ndarray, B: np.ndarray, top_k=3):
    """Try all axis permutations and sign flips on A to find the mapping that
    minimizes mean angular error to B. Returns a list of top_k tuples
    (mean_angle, perm, signs).
    """
    import itertools
    best = []
    # ensure arrays
    A = np.asarray(A)
    B = np.asarray(B)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            # apply permutation and signs
            A_perm = A[:, perm] * np.array(signs)[None, :]
            mean_val, _ = mean_angle_between_sets(A_perm, B)
            best.append((mean_val, perm, signs))
    best.sort(key=lambda x: x[0])
    return best[:top_k]

def feature_separation_loss(f1, f2, epsilon=1e-6):
    """Encourage decorrelation / orthogonality between f1 and f2.
    在浮点32中进行，减少半精度下归一化误差。"""
    f1f = f1.float()
    f2f = f2.float()
    f1_n = f1f / (f1f.norm(dim=-1, keepdim=True) + epsilon)
    f2_n = f2f / (f2f.norm(dim=-1, keepdim=True) + epsilon)
    cos = (f1_n * f2_n).sum(dim=-1).clamp(-1.0, 1.0)
    return cos.abs().mean()
def angular_loss(pred, target):
    pf = pred.float()
    tf = target.float()
    pred_n = pf / (pf.norm(dim=-1, keepdim=True) + 1e-6)
    target_n = tf / (tf.norm(dim=-1, keepdim=True) + 1e-6)
    cos_sim = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # 构造数据集，这里使用 ZhaoDataset 包裹 DatasetMPIIFaceGazeByGazeHub
    # TODO 多折验证。这里只验证了第一折 Chen-Xianrui
    #for eyediap and mpiiface
    # train_ds = ZhaoDataset(leave_one_out(TRAIN_DATASET_NAME, TRAIN_LABELS_PATH, 0), TRAIN_DATASET_NAME)
    # test_ds = ZhaoDataset(one(TEST_DATASET_NAME, TEST_LABELS_PATH, 0), TEST_DATASET_NAME)
    
    # -----------------------------
    # 数据集与域适应设置
    # 需求1: 当使用 ETH-XGaze 作为训练集时，使用全部 train.label 训练；测试集来自 config 中指定的另一数据集 (域适应场景)
    # -----------------------------
    DOMAIN_ADAPTATION = True  # 标志是否开启对抗域适应
    if TRAIN_DATASET_NAME == "ETH-XGaze":
        # 使用全部 train.label
        train_label_path = [TRAIN_LABELS_PATH / "train.label"]
        train_ds = ZhaoDataset(TRAIN_IMAGES_PATH, train_label_path, TRAIN_DATASET_NAME)
        print(f"{TRAIN_DATASET_NAME} 全部训练样本数: {len(train_ds)}")

        # 构建目标域（测试域）数据集: 使用 TEST_DATASET_NAME 指定的数据集所有标签文件
        # 收集所有标签文件 (p*.label / *.label)
        all_test_label_paths = []
        if TEST_DATASET_NAME == "MPIIFaceGaze" or TEST_DATASET_NAME == "EyeDiap":
            all_test_label_paths = sorted([p for p in TEST_LABELS_PATH.glob("p*.label")])
        elif TEST_DATASET_NAME == "Gaze360":
            # Gaze360 使用 train/test.label 命名, 若作为 target domain, 默认使用其 train.label + test.label
            cand = [TEST_LABELS_PATH / "train.label", TEST_LABELS_PATH / "test.label"]
            all_test_label_paths = [p for p in cand if p.exists()]
        elif TEST_DATASET_NAME == "ETH-XGaze":
            all_test_label_paths = [TEST_LABELS_PATH / "test.label"]
        else:
            # 回退策略: 所有 .label
            all_test_label_paths = list(TEST_LABELS_PATH.glob("*.label"))

        if len(all_test_label_paths) == 0:
            raise RuntimeError(f"未找到目标域标签文件: {TEST_LABELS_PATH}")
        test_ds = ZhaoDataset(TEST_IMAGES_PATH, all_test_label_paths, TEST_DATASET_NAME)
        print(f"目标域 {TEST_DATASET_NAME} 样本数: {len(test_ds)}")

        gen = torch.Generator().manual_seed(SEED)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=gen,
            num_workers=NUM_WORKERS,
        )
        # 评估用 DataLoader (不打乱)
        test_dl = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        # 额外的目标域 DataLoader 供域适应训练 (打乱)
        target_domain_dl = DataLoader(
            TargetOnlyDataset(test_ds),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        DOMAIN_ADAPTATION = True  # 域适应开启
    elif TRAIN_DATASET_NAME == "Gaze360":
        # 现在需求更新：不再从 test/Face 读取映射图片，直接使用 train.label 中的原始路径 (train/Face/...)
        # 训练集：Gaze360 train.label + 原始路径
        gaze360_train_label = [TRAIN_LABELS_PATH / "train.label"]
        train_ds = ZhaoDataset(TRAIN_IMAGES_PATH, gaze360_train_label, "Gaze360")
        # 测试 / 目标域：EyeDiap p1..p16.label + p*/face 图像
        eyediap_label_paths = sorted([p for p in TEST_LABELS_PATH.glob("p*.label")])
        if len(eyediap_label_paths) == 0:
            raise RuntimeError(f"未找到 EyeDiap 测试标签文件: {TEST_LABELS_PATH}")
        test_ds = DatasetEyeDiapByGazeHub(
            TEST_IMAGES_PATH, eyediap_label_paths
        )
        # 为了评估稳定性，去除目标域(测试域)上的随机数据增强（随机裁剪、翻转、颜色抖动），改为纯 ToTensor+Normalize
        try:
            from torchvision import transforms as _t
            eval_transform = _t.Compose([
                _t.ToTensor(),
                _t.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
            # 覆盖 test_ds 的 transform 与 other_transform（EyeDiap 类结构里这两个属性在实例化后可直接替换）
            if hasattr(test_ds, 'transform'):
                test_ds.transform = eval_transform
            if hasattr(test_ds, 'other_transform'):
                test_ds.other_transform = eval_transform
            print("[INFO] Disabled random augmentation for EyeDiap test domain (deterministic eval transforms).")
        except Exception as _e_aug:
            print(f"[WARN] Failed to override test transforms: {_e_aug}")
        print(f"Gaze360(train.label 原路径) 训练样本数: {len(train_ds)}, EyeDiap 测试样本数: {len(test_ds)}")
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        # For non-ETH-XGaze setups we may still want a target-domain iterator
        # for domain-adaptation training. Create a TargetOnly DataLoader from
        # the test set so later code can always rely on `target_domain_dl`.
        if DOMAIN_ADAPTATION:
            target_domain_dl = DataLoader(
                TargetOnlyDataset(test_ds),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
            )
        else:
            target_domain_dl = None
    else:
        train_label_path = leave_one_out(TRAIN_DATASET_NAME, TRAIN_LABELS_PATH, 1)
        val_label_path = one(TEST_DATASET_NAME, TEST_LABELS_PATH, 1)
        train_ds = ZhaoDataset(TRAIN_IMAGES_PATH, train_label_path, TRAIN_DATASET_NAME)
        test_ds = ZhaoDataset(TEST_IMAGES_PATH, val_label_path, TRAIN_DATASET_NAME)
        print("训练集样本数:", len(train_ds))
        print("测试集样本数:", len(test_ds))
        gen = torch.Generator().manual_seed(SEED)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=gen,
            num_workers=NUM_WORKERS,
        )
        print("训练集 batch 数量:", len(train_dl))
        test_dl = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        # Ensure target_domain_dl exists for downstream domain-adaptation logic
        if DOMAIN_ADAPTATION:
            target_domain_dl = DataLoader(
                TargetOnlyDataset(test_ds),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
            )
        else:
            target_domain_dl = None

    # Ensure `target_domain_dl` exists so later iteration can't NameError.
    # Some dataset branches may have failed to set it; default to None.
    if 'target_domain_dl' not in locals():
        target_domain_dl = None

    # 加载主模型（例如 CLIP 模型）到 GPU
    model = GEWithCLIPModel().to(DEVICE)
    for param in model.model.parameters():
        param.requires_grad = False
    model.model.float()
    # model.main_model = model.main_model.half()  # 如果 main_model 是 CNN

    # 实例化 TransformerDeepSeek_gaze 模型，注意内部投影层参数维度需跟数据一致：
    config = TransformerConfigBase()
    transformer_model = TransformerDeepSeek_gaze(
        config.num_layers,
        config.embed_dim,
        config.inter_dim,
        config.num_heads,
        config.n_routed_experts,
        config.n_activated_experts,
        config.n_shared_experts,
        config.moe_inter_dim,
        d_model=768,  # 统一投影到 768 维
        out_dim=3  # gaze 输出维度为 3
    ).to(DEVICE)

    # -----------------------------
    # 域适应: 定义梯度反转层 & 域判别器
    # -----------------------------
    class GradReverseFn(Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)
        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.alpha * grad_output, None
    def grad_reverse(x, alpha=1.0):
        return GradReverseFn.apply(x, alpha)

    # We'll lazily create a domain discriminator when we know the exact concatenated feature dim
    domain_discriminator = None
    domain_projector = None
    domain_optimizer = None

    # -----------------------------
    # 判别器稳定化/防饱和超参数
    # -----------------------------
    # 加强判别器: 提升学习率系数、去掉熵混淆、降低/关闭标签平滑、提高反转强度上限
    DISC_LR_FACTOR = 0.5          # 提升判别器学习率（相对主网络）
    DISC_LABEL_SMOOTH = 0.0       # 关闭标签平滑让其更快收敛
    DISC_ALPHA_MAX = 0.6          # 允许更强的梯度反转（但仍低于1以防过早震荡）
    DISC_ENTROPY_WEIGHT = 0.1     # 关闭熵正则
    DISC_UPDATE_EVERY = 1         # 每步更新
    USE_ENTROPY_CONFUSION = False # 不再使用熵混淆
    # Prototype alignment removed in adversarial-only pruning
    
    trainable_params = (
        list(transformer_model.parameters())
        + list(model.semantic_parameters())
        + list(model.fuse_model.parameters())
        + list(model.main_model.parameters())
    )

    # 主优化器不包含判别器的参数；判别器单独优化以稳定对抗训练
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )
    # domain_optimizer will be created when domain_discriminator is instantiated on first batch
    domain_optimizer = None

    def cross_entropy_with_smoothing(logits, targets, smoothing):
        if smoothing <= 0:
            return F.cross_entropy(logits, targets)
        # 使用 torch>=1.10 支持的 label_smoothing 参数 (当前环境 torch 2.x)
        return F.cross_entropy(logits, targets, label_smoothing=smoothing)

    # 添加余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=ETA_MIN
    )

    # Mean-Teacher and prototype alignment removed for a focused adversarial training script.


    os.makedirs("log", exist_ok=True)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"log/{log_time}_{TRAIN_DATASET_NAME}-{TEST_DATASET_NAME}_log.txt"


    def write_log(msg):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


    # Evaluation helper: run a quick evaluation on a DataLoader (limited batches)
    def evaluate_on_dataloader(transformer_model, model, data_loader, max_batches=10, use_amp=True):
        """Compute mean angular error (degrees) over up to max_batches from data_loader.
        Returns mean_angle (float) or float('nan') on empty loader.
        Added diagnostics: collects a small number of per-sample preds/gt for debugging.
        """
        transformer_model.eval()
        model.eval()
        total_angle = 0.0
        n_samples = 0
        # diagnostics
        collected_preds = []
        collected_gts = []
        max_debug_samples = 0
        with torch.no_grad():
            for bi, batch in enumerate(data_loader):
                if bi >= max_batches:
                    break
                with autocast(enabled=use_amp):
                    raw_inputs = process_batch(batch, model=model)
                    # ensure labels are float for loss/angle computation
                    for k in raw_inputs:
                        if isinstance(raw_inputs[k], torch.Tensor):
                            raw_inputs[k] = raw_inputs[k].float()
                    output = transformer_model(raw_inputs)
                    output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                    label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                    pred_n = output_safe / (output_safe.norm(dim=-1, keepdim=True) + 1e-6)
                    gt_n = label_safe / (label_safe.norm(dim=-1, keepdim=True) + 1e-6)
                    cos_sim = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
                    angles = torch.acos(cos_sim) * 180.0 / math.pi
                    total_angle += angles.sum().item()
                    n_samples += angles.numel()
                    # collect a few sample preds/gts for debugging (store as float numpy arrays)
                    if len(collected_preds) < 20:
                        try:
                            preds_cpu = pred_n.detach().cpu().numpy()
                            gts_cpu = gt_n.detach().cpu().numpy()
                            for pi in range(preds_cpu.shape[0]):
                                if len(collected_preds) >= 20:
                                    break
                                collected_preds.append(preds_cpu[pi].astype(float))
                                collected_gts.append(gts_cpu[pi].astype(float))
                        except Exception:
                            pass
        if n_samples == 0:
            return float('nan')
        mean_angle = total_angle / n_samples
        # attach diagnostics to return via attributes on function (non-intrusive)
        evaluate_on_dataloader._last_debug_preds = collected_preds
        evaluate_on_dataloader._last_debug_gts = collected_gts
        return mean_angle

    # -----------------------------
    # 评估函数：支持 step 级别调用 & 自动整体翻转检测 (pred -> -pred)
    # -----------------------------
    # Evaluation helper removed to streamline the adversarial training script.
    
    # step 级评估间隔
    # 缩短评估间隔，便于在跨域高误差情况下更快观察趋势
    EVAL_INTERVAL_STEPS = 1000
    # 每轮从目标域中抽取若干有标签样本用于有监督域适应训练
    USE_TARGET_SUPERVISED = True
    TARGET_SUP_NUM = 100  # 每轮抽取不同的目标域样本数量
    TARGET_SUP_WEIGHT = 1.0  # 有监督目标样本损失权重
    # 伪标签（self-training）超参数
    USE_PSEUDO_LABELS = True
    PSEUDO_CONF_THRESHOLD = 0.85  # 置信度阈值（余弦相似度/概率）用于筛选伪标签
    PSEUDO_MAX_PER_EPOCH = 500  # 每轮允许的最大伪标签样本数
    PSEUDO_WEIGHT = 0.5  # 伪标签损失权重
    PSEUDO_APPLY_EVERY_K_STEPS = 10  # 每 K 步应用一次伪标签批


    os.makedirs("checkpoints", exist_ok=True)
    best_angle = float('inf')
    best_model_path = None
    # AMP 设置：原先在域适应阶段关闭 AMP（为避免 dtype 混用问题），
    # 但你希望在域适应阶段也启用混合精度以节省显存并加速训练。
    # 我们启用 AMP（autocast + GradScaler），同时强制把所有训练/前向相关模块设置为 float32，
    # 以避免出现 fp16 参数导致的 unscale/数值问题。
    use_amp = True
    scaler = GradScaler() if use_amp else None
    # 若未启用 AMP，可手动转为 float32；启用 AMP 时保持原始精度以便半精度加速
    # 统一策略：使用 AMP 时也保证参数为 float32（自动混合精度只影响计算，不影响参数）；
    # 未使用 AMP 也保持 float32
    for _m in [transformer_model, getattr(model, 'main_model', None), getattr(model, 'fuse_model', None)]:
        try:
            if _m is not None:
                _m.float()
        except Exception:
            pass

    # 打印 dtype 检查，帮助确认所有关键模块为 float32，以及 AMP 状态
    def print_module_dtype(name, module):
        try:
            d = next(module.parameters()).dtype
        except Exception:
            d = None
        print(f"[DTypeCheck] {name} param dtype: {d}")

    print(f"[AMP] use_amp={use_amp}, scaler={'enabled' if scaler is not None else 'disabled'}")
    print_module_dtype('CLIP model.model', model.model)
    try:
        print_module_dtype('CLIP visual', model.model.visual)
    except Exception:
        pass
    print_module_dtype('transformer_model', transformer_model)
    try:
        print_module_dtype('model.main_model', model.main_model)
    except Exception:
        pass
    try:
        print('[DTypeCheck] desc prototype dtype:', model.desc_feats.dtype)
    except Exception:
        pass
    global_step = 0
    # 训练开始前的基线评估（不带任何梯度更新），帮助确认初始跨域误差水平
    try:
        base_angle = evaluate_on_dataloader(transformer_model, model, test_dl, max_batches=20, use_amp=True)
        base_msg = f"[BASELINE] Pre-train cross-domain MeanAngle={base_angle:.4f}° (first 20 batches of target)"
        print(base_msg)
        write_log(base_msg)
        # diagnostic: if small sample preds/gts were collected, compute Kabsch alignment
        try:
            preds = evaluate_on_dataloader._last_debug_preds
            gts = evaluate_on_dataloader._last_debug_gts
            if preds and gts and len(preds) == len(gts):
                preds_arr = np.vstack(preds)
                gts_arr = np.vstack(gts)
                mean_before, _ = mean_angle_between_sets(preds_arr, gts_arr)
                R, preds_aligned = kabsch_rotation(preds_arr, gts_arr)
                mean_after, _ = mean_angle_between_sets(preds_aligned, gts_arr)
                diag_msg = f"[BASELINE_DIAG] samples={len(preds_arr)} mean_before={mean_before:.4f}° mean_after={mean_after:.4f}°"
                print(diag_msg)
                write_log(diag_msg)
                # Removed verbose per-sample and permutation debug output to reduce log noise.
        except Exception as _d_e:
            print(f"[BASELINE_DIAG_ERR] {_d_e}")
    except Exception as _e_base:
        print(f"[BASELINE][ERR] {str(_e_base)}")
    # 如果启用跨 epoch 不重复抽样，预生成一次全局 permutation 并维护指针
    # Prepare fixed supervised target subset (same indices every epoch) and disjoint eval subset
    if DOMAIN_ADAPTATION and USE_TARGET_SUPERVISED:
        gen_global = torch.Generator().manual_seed(SEED)
        target_sup_perm = torch.randperm(len(test_ds), generator=gen_global).tolist()
        # decide how many supervised target samples we'll actually use (cap by dataset size)
        n_target_sup = min(TARGET_SUP_NUM, len(test_ds))
        sup_indices = target_sup_perm[:n_target_sup]
        # build a fixed supervised DataLoader (same samples every epoch)
        target_sup_subset = Subset(test_ds, sup_indices)
        target_sup_dl = DataLoader(target_sup_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        target_sup_iter = iter(target_sup_dl)
        # build a disjoint evaluation subset (complement of supervised indices)
        sup_set = set(sup_indices)
        all_idx = set(range(len(test_ds)))
        eval_indices = sorted(list(all_idx - sup_set))
        if len(eval_indices) == 0:
            target_eval_dl = test_dl
            warn_msg = "[WARN] No disjoint eval subset (all target labeled used for supervision); evaluation will use full test_dl."
            print(warn_msg)
            write_log(warn_msg)
        else:
            target_eval_subset = Subset(test_ds, eval_indices)
            target_eval_dl = DataLoader(target_eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            split_msg = f"[SPLIT_FIXED] SupervisedTarget={len(sup_indices)} EvalSubset={len(eval_indices)} TotalTarget={len(test_ds)}"
            print(split_msg)
            write_log(split_msg)
        # persist supervised indices for reproducibility
        try:
            idx_path = os.path.join("checkpoints", f"fixed_target_sup_indices_seed{SEED}_n{n_target_sup}.txt")
            with open(idx_path, "w", encoding="utf-8") as _f:
                _f.write("\n".join([str(int(x)) for x in sup_indices]))
            write_log(f"[SAVED] fixed supervised target indices -> {idx_path}")
        except Exception as e:
            write_log(f"[ERR] Failed to save sup indices: {e}")

    # 轻量伪标签封装
    class PseudoLabelDataset(Dataset):
        def __init__(self, base_dataset, pseudo_labels):
            # pseudo_labels: dict index -> label_tensor
            self.base = base_dataset
            self.pseudo = pseudo_labels
            self.indices = sorted(list(pseudo_labels.keys()))
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            inp, _ = self.base[real_idx]
            label = self.pseudo[real_idx]
            return inp, label

    for epoch in range(NUM_EPOCHS):
        # --- 训练 ---
        if DOMAIN_ADAPTATION:
            target_iter = iter(target_domain_dl)
            # Use fixed supervised target subset created before the training loop.
            # `target_sup_dl` and `target_eval_dl` are constructed once at startup and reused each epoch.
            if USE_TARGET_SUPERVISED:
                pass
        LOG_INTERVAL = 50  # 原来是每50步，现在缩短，便于观察
        import time
        batch_start_time = time.time()
        for i, batch in enumerate(train_dl):
            # 先从迭代器取出 source/target raw batches（注意不要预先在外部做大量前向计算）
            domain_loss_det_cached = None  # 每步初始化缓存判别器损失
            if DOMAIN_ADAPTATION:
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_domain_dl)
                    target_batch = next(target_iter)
            else:
                target_batch = None

            # 从本轮的有监督目标子集中取一批样本（若启用）
            if DOMAIN_ADAPTATION and USE_TARGET_SUPERVISED:
                try:
                    target_sup_batch = next(target_sup_iter)
                except Exception:
                    # 重新创建迭代器以循环使用（若子集小于训练步数）
                    target_sup_iter = iter(target_sup_dl)
                    target_sup_batch = next(target_sup_iter)
            else:
                target_sup_batch = None

            # 每步初始化目标子集监督损失的打印值
            sup_loss_val = float('nan')
            # 把 process_batch 与前向计算放到 autocast 上下文中，这样 AMP 才能覆盖主要计算开销
            with autocast(enabled=use_amp):
                raw_inputs = process_batch(batch, model=model)  # 源域 batch (在 autocast 中执行)
                # 训练前监测源域特征
                debug_check_tensor("src_feature_1_pre", raw_inputs.get("feature_1"))
                debug_check_tensor("src_feature_3_pre", raw_inputs.get("feature_3"))

                raw_inputs_t = None
                if DOMAIN_ADAPTATION and target_batch is not None:
                    # target batch wrapped by TargetOnlyDataset so labels are dummy
                    raw_inputs_t = process_batch(target_batch, model=model)
                    debug_check_tensor("tgt_feature_1_pre", raw_inputs_t.get("feature_1"))
                    debug_check_tensor("tgt_feature_3_pre", raw_inputs_t.get("feature_3"))

                # 不再对全部特征强制 float32，只对 label / loss 内部再转换；保留半精度带来的显存与吞吐收益
                if target_sup_batch is not None:
                    raw_sup = process_batch(target_sup_batch, model=model)

                output = transformer_model(raw_inputs)
                output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                loss_gaze = angular_loss(output_safe, label_safe)
                loss_sep = feature_separation_loss(raw_inputs["feature_1"], raw_inputs["feature_2"])
                lambda_sep = 1.0 if DOMAIN_ADAPTATION else 0.0
                total_loss = loss_gaze + lambda_sep * loss_sep

                # 若启用，计算目标子集的监督损失并加入总损失（有标签的目标域样本用于有监督适配）
                if target_sup_batch is not None:
                    sup_output = transformer_model(raw_sup)
                    sup_output_safe = torch.nan_to_num(sup_output, nan=0.0, posinf=1e6, neginf=-1e6)
                    sup_label_safe = torch.nan_to_num(raw_sup["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                    sup_loss_gaze = angular_loss(sup_output_safe, sup_label_safe)
                    total_loss = total_loss + TARGET_SUP_WEIGHT * sup_loss_gaze
                    # 记录可打印的数值以供步级日志使用（脱离 AMP/设备）
                    try:
                        sup_loss_val = float(sup_loss_gaze.detach().cpu().item())
                    except Exception:
                        sup_loss_val = float('nan')

                domain_loss_val = torch.tensor(0.0, device=DEVICE)
                if DOMAIN_ADAPTATION:
                    # 提取域特征 (feature_1 + 平均池化后的 CNN token)
                    def domain_feat(ri):
                        f1 = ri["feature_1"]  # [B,512]
                        f3_tokens = ri["feature_3"]  # [B, N, 2048]
                        f2 = ri.get("feature_2")
                        f4 = ri.get("token_img_patch")
                        if f3_tokens.dim() == 3:
                            f3 = f3_tokens.mean(dim=1)
                        else:
                            # 兼容异常情况
                            f3 = f3_tokens.view(f1.size(0), -1)
                        # flatten/avg f2 if it's a sequence
                        if isinstance(f2, torch.Tensor):
                            if f2.dim() == 2:
                                f2_vec = f2
                            else:
                                f2_vec = f2.mean(dim=1)
                        else:
                            f2_vec = torch.zeros_like(f1)
                        # f4 may be None or a token sequence
                        if isinstance(f4, torch.Tensor):
                            if f4.dim() == 3:
                                f4_vec = f4.mean(dim=1)
                            else:
                                f4_vec = f4.view(f1.size(0), -1)
                        else:
                            f4_vec = torch.zeros_like(f1)
                        debug_check_tensor("domain_feat_f1", f1)
                        debug_check_tensor("domain_feat_f3", f3)
                        # concat full set: f1, f2_vec, f3, f4_vec
                        cat = torch.cat([f1, f2_vec, f3, f4_vec], dim=-1)
                        # optional small projection to stabilize dim
                        global domain_projector, domain_discriminator, domain_optimizer
                        if domain_projector is None:
                            D_in = cat.size(-1)
                            proj_hidden = max(256, min(1024, D_in // 4))
                            domain_projector = torch.nn.Sequential(
                                torch.nn.Linear(D_in, proj_hidden),
                                torch.nn.LayerNorm(proj_hidden),
                                torch.nn.ReLU()
                            ).to(DEVICE)
                        proj = domain_projector(cat)
                        # normalize projector outputs to stabilize discriminator and distance-based losses
                        proj = F.normalize(proj, dim=-1)
                        # create discriminator lazily
                        if domain_discriminator is None:
                            domain_discriminator = torch.nn.Sequential(
                                torch.nn.Linear(proj.size(-1), 256),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.3),
                                torch.nn.Linear(256, 2)
                            ).to(DEVICE)
                            domain_optimizer = torch.optim.AdamW(domain_discriminator.parameters(), lr=LEARNING_RATE * DISC_LR_FACTOR)
                        return proj
                    feat_s = domain_feat(raw_inputs)
                    feat_t = domain_feat(raw_inputs_t)
                    # Prototype alignment removed (adversarial-only pruning)
                    # 记录判别器预测用于计算准确率
                    with torch.no_grad():
                        ds_logits_eval = domain_discriminator(torch.cat([feat_s, feat_s * 0.0], dim=0)) if False else None
                    # 计算 alpha (随训练进度动态调整)
                    p = float(epoch * len(train_dl) + i) / (NUM_EPOCHS * len(train_dl) + 1e-6)
                    alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
                    # 限制最大 alpha，避免过早过强的梯度反转
                    alpha = min(alpha, DISC_ALPHA_MAX)
                    feat_s_grl = grad_reverse(feat_s, alpha)
                    feat_t_grl = grad_reverse(feat_t, alpha)
                    # First: update discriminator on detached features (no GRL)
                    disc_acc = float('nan')  # default
                    feat_s_det = feat_s.detach()
                    feat_t_det = feat_t.detach()
                    # 判别器可选择降低更新频率
                    if (i % DISC_UPDATE_EVERY) == 0:
                        # 仅计算判别器 detach logits & 损失，缓存到 domain_loss_det_cached，统一在后面 backward
                        ds_logits_det = domain_discriminator(feat_s_det)
                        dt_logits_det = domain_discriminator(feat_t_det)
                        domain_labels_s = torch.zeros(ds_logits_det.size(0), dtype=torch.long, device=DEVICE)
                        domain_labels_t = torch.ones(dt_logits_det.size(0), dtype=torch.long, device=DEVICE)
                        domain_loss_s_det = cross_entropy_with_smoothing(ds_logits_det, domain_labels_s, DISC_LABEL_SMOOTH)
                        domain_loss_t_det = cross_entropy_with_smoothing(dt_logits_det, domain_labels_t, DISC_LABEL_SMOOTH)
                        domain_loss_det_cached = 0.5 * (domain_loss_s_det + domain_loss_t_det)
                        if USE_ENTROPY_CONFUSION:
                            probs_s_det = F.softmax(ds_logits_det, dim=-1)
                            probs_t_det = F.softmax(dt_logits_det, dim=-1)
                            entropy_s = -(probs_s_det * (probs_s_det.clamp_min(1e-6).log())).sum(dim=-1).mean()
                            entropy_t = -(probs_t_det * (probs_t_det.clamp_min(1e-6).log())).sum(dim=-1).mean()
                            entropy_mean = 0.5 * (entropy_s + entropy_t)
                            domain_loss_det_cached = domain_loss_det_cached - DISC_ENTROPY_WEIGHT * entropy_mean
                        else:
                            entropy_mean = torch.tensor(0.0, device=DEVICE)
                        with torch.no_grad():
                            preds_s = ds_logits_det.argmax(dim=1)
                            preds_t = dt_logits_det.argmax(dim=1)
                            acc_s = (preds_s == domain_labels_s).float().mean().item()
                            acc_t = (preds_t == domain_labels_t).float().mean().item()
                            disc_acc = 0.5 * (acc_s + acc_t)
                            disc_entropy = entropy_mean.item() if USE_ENTROPY_CONFUSION else float('nan')
                    else:
                        disc_acc = float('nan')
                        disc_entropy = float('nan')

                    # Then compute adversarial loss for main model using GRL outputs
                    ds_logits = domain_discriminator(feat_s_grl)
                    dt_logits = domain_discriminator(feat_t_grl)
                    debug_check_tensor("ds_logits", ds_logits)
                    debug_check_tensor("dt_logits", dt_logits)
                    domain_labels_s_main = torch.zeros(ds_logits.size(0), dtype=torch.long, device=DEVICE)
                    domain_labels_t_main = torch.ones(dt_logits.size(0), dtype=torch.long, device=DEVICE)
                    domain_loss_s = F.cross_entropy(ds_logits, domain_labels_s_main)
                    domain_loss_t = F.cross_entropy(dt_logits, domain_labels_t_main)
                    domain_loss_val = 0.5 * (domain_loss_s + domain_loss_t)
                    lambda_domain = 1.0  # 可调权重
                    total_loss = total_loss + lambda_domain * domain_loss_val

                    # Mean-Teacher removed in streamlined script

                pred_n = output_safe / (output_safe.norm(dim=-1, keepdim=True) + 1e-6)
                gt_n = label_safe / (label_safe.norm(dim=-1, keepdim=True) + 1e-6)
                cos_sim = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
                angles = torch.acos(cos_sim) * 180.0 / math.pi
                mean_angle = angles.mean().item()
            # 如果出现 NaN，跳过该 batch
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[WARN] Skip batch {i} due to NaN/Inf in loss.")
                optimizer.zero_grad(set_to_none=True)
                continue
            if (i % LOG_INTERVAL) == 0:
                elapsed = time.time() - batch_start_time
                batch_start_time = time.time()
                if DOMAIN_ADAPTATION:
                    log_msg = (
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}——{TRAIN_DATASET_NAME}->{TEST_DATASET_NAME} "
                        f"Epoch {epoch} Step {i}, TotalLoss: {total_loss.item():.6f}, Gaze: {loss_gaze.item():.6f}, "
                        f"FeatSep: {loss_sep.item():.6f}, SupTarget: {sup_loss_val if not math.isnan(sup_loss_val) else 'nan'}, "
                        f"Domain: {domain_loss_val.item():.6f}, DiscAcc: {disc_acc if not isinstance(disc_acc,float) or not math.isnan(disc_acc) else float('nan'):.3f}, "
                        f"Alpha: {alpha:.3f}, TrainBatchAngle: {mean_angle:.2f}°, BatchTime: {elapsed:.2f}s"
                    )
                else:
                    log_msg = (
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}——{TRAIN_DATASET_NAME}-{TEST_DATASET_NAME} "
                        f"Epoch {epoch} Step {i}, Loss: {total_loss.item():.6f}, Angular Loss: {loss_gaze.item():.6f}, "
                        f"Feature Loss: {loss_sep.item():.6f}, Mean Angular Error: {mean_angle:.2f}°, BatchTime: {elapsed:.2f}s"
                    )
                print(log_msg)
                write_log(log_msg)
            # =========================
            # Backward & Optimizer steps (Reworked for single scaler.update)
            # =========================
            if use_amp:
                # 重新组织：提前 zero_grad，累积两个损失再一步更新
                if domain_optimizer is not None:
                    domain_optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                # 1) 判别器 detached 损失（若存在）
                if DOMAIN_ADAPTATION and 'domain_loss_det' in locals():
                    # domain_loss_det 在上方定义时未保存；改为临时变量记录
                    pass  # 保持兼容，不做操作
                # 判别器独立损失需要重新计算（避免之前已 step）
                domain_loss_det_cached = None
                if DOMAIN_ADAPTATION and domain_discriminator is not None:
                    # 重新计算判别器 detach forward（避免使用可能被覆盖的变量）
                    with autocast(enabled=use_amp):
                        feat_s_det2 = feat_s.detach()
                        feat_t_det2 = feat_t.detach()
                        ds_logits_det2 = domain_discriminator(feat_s_det2)
                        dt_logits_det2 = domain_discriminator(feat_t_det2)
                        domain_labels_s2 = torch.zeros(ds_logits_det2.size(0), dtype=torch.long, device=DEVICE)
                        domain_labels_t2 = torch.ones(dt_logits_det2.size(0), dtype=torch.long, device=DEVICE)
                        domain_loss_s_det2 = cross_entropy_with_smoothing(ds_logits_det2, domain_labels_s2, DISC_LABEL_SMOOTH)
                        domain_loss_t_det2 = cross_entropy_with_smoothing(dt_logits_det2, domain_labels_t2, DISC_LABEL_SMOOTH)
                        domain_loss_det_cached = 0.5 * (domain_loss_s_det2 + domain_loss_t_det2)
                with autocast(enabled=use_amp):
                    main_total_loss = total_loss  # 已包含 adversarial domain_loss_val
                if domain_loss_det_cached is not None:
                    scaler.scale(domain_loss_det_cached).backward(retain_graph=True)
                scaler.scale(main_total_loss).backward()
                # Unscale & clip
                if domain_optimizer is not None:
                    scaler.unscale_(domain_optimizer)
                    torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=5.0)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
                # Steps
                if domain_optimizer is not None:
                    scaler.step(domain_optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                if domain_optimizer is not None:
                    domain_optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                # 判别器 detach 再算一次
                domain_loss_det_cached = None
                if DOMAIN_ADAPTATION and domain_discriminator is not None:
                    feat_s_det2 = feat_s.detach()
                    feat_t_det2 = feat_t.detach()
                    ds_logits_det2 = domain_discriminator(feat_s_det2)
                    dt_logits_det2 = domain_discriminator(feat_t_det2)
                    domain_labels_s2 = torch.zeros(ds_logits_det2.size(0), dtype=torch.long, device=DEVICE)
                    domain_labels_t2 = torch.ones(dt_logits_det2.size(0), dtype=torch.long, device=DEVICE)
                    domain_loss_s_det2 = cross_entropy_with_smoothing(ds_logits_det2, domain_labels_s2, DISC_LABEL_SMOOTH)
                    domain_loss_t_det2 = cross_entropy_with_smoothing(dt_logits_det2, domain_labels_t2, DISC_LABEL_SMOOTH)
                    domain_loss_det_cached = 0.5 * (domain_loss_s_det2 + domain_loss_t_det2)
                if domain_loss_det_cached is not None:
                    domain_loss_det_cached.backward(retain_graph=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
                if domain_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), max_norm=5.0)
                    domain_optimizer.step()
                optimizer.step()

            global_step += 1
            # Periodic evaluation on small supervised target subset
            if DOMAIN_ADAPTATION and USE_TARGET_SUPERVISED and (global_step % EVAL_INTERVAL_STEPS == 0):
                try:
                    # 优先使用与监督样本不重叠的 target_eval_dl
                    eval_loader = target_eval_dl if 'target_eval_dl' in locals() else None
                    if eval_loader is None:
                        # 回退到完整 test_dl（仍然是合法评估，只是包含已监督的样本或未划分情况）
                        eval_loader = test_dl if 'test_dl' in locals() else None
                    if eval_loader is not None:
                        mean_angle = evaluate_on_dataloader(transformer_model, model, eval_loader, max_batches=20, use_amp=use_amp)
                        eval_msg = f"[EVAL] Step {global_step} MeanAngle={mean_angle:.4f}° (disjoint target eval subset)"
                        print(eval_msg)
                        write_log(eval_msg)
                        # save checkpoint if improved
                        if mean_angle < best_angle:
                            best_angle = mean_angle
                            if not is_ablation:
                                ckpt_path = os.path.join("checkpoints", f"best—step{global_step}_angle{best_angle:.4f}.pt")
                                try:
                                    torch.save({
                                        'epoch': epoch,
                                        'global_step': global_step,
                                        'transformer_state': transformer_model.state_dict(),
                                        'clip_model_state': model.state_dict(),
                                        'optimizer_state': optimizer.state_dict(),
                                        'scheduler_state': scheduler.state_dict(),
                                        'best_angle': best_angle,
                                    }, ckpt_path)
                                    best_model_path = ckpt_path
                                    save_msg = f"[CKPT] Saved improved checkpoint to {ckpt_path}"
                                    print(save_msg)
                                    write_log(save_msg)
                                except Exception as e:
                                    err_msg = f"[ERR] Failed to save checkpoint: {e}"
                                    print(err_msg)
                                    write_log(err_msg)
                except Exception as e:
                    err_msg = f"[EVAL_ERR] evaluation failed at step {global_step}: {e}"
                    print(err_msg)
                    write_log(err_msg)
        # ---- Epoch end ----
        transformer_model.train()
        model.train()
        scheduler.step()  # 现在放在 epoch 内部，确保每轮更新
        current_lr = scheduler.get_last_lr()[0]
        lr_log = f"[LR] Epoch {epoch} LR={current_lr:.6e}"
        print(lr_log)
        write_log(lr_log)
        # epoch 级快速评估
        try:
            eval_loader_ep = None
            if DOMAIN_ADAPTATION and 'target_eval_dl' in locals():
                eval_loader_ep = target_eval_dl
            elif 'test_dl' in locals():
                eval_loader_ep = test_dl
            if eval_loader_ep is not None:
                ep_angle = evaluate_on_dataloader(transformer_model, model, eval_loader_ep, max_batches=40, use_amp=True)
                ep_msg = f"[EPOCH_EVAL] Epoch {epoch} MeanAngle={ep_angle:.4f}° (target domain quick eval)"
                print(ep_msg)
                write_log(ep_msg)
                # diagnostic: Kabsch align on collected small sample set
                try:
                    preds = evaluate_on_dataloader._last_debug_preds
                    gts = evaluate_on_dataloader._last_debug_gts
                    if preds and gts and len(preds) == len(gts):
                        preds_arr = np.vstack(preds)
                        gts_arr = np.vstack(gts)
                        mean_before, _ = mean_angle_between_sets(preds_arr, gts_arr)
                        R, preds_aligned = kabsch_rotation(preds_arr, gts_arr)
                        mean_after, _ = mean_angle_between_sets(preds_aligned, gts_arr)
                        diag_msg = f"[EPOCH_DIAG] Epoch {epoch} samples={len(preds_arr)} mean_before={mean_before:.4f}° mean_after={mean_after:.4f}°"
                        print(diag_msg)
                        write_log(diag_msg)
                except Exception as _d_e:
                    print(f"[EPOCH_DIAG_ERR] {_d_e}")
                if ep_angle < best_angle:
                    best_angle = ep_angle
                    if not is_ablation:
                        ckpt_path = os.path.join("checkpoints", f"best—epoch{epoch}_angle{best_angle:.4f}.pt")
                        try:
                            torch.save({
                                'epoch': epoch,
                                'global_step': global_step,
                                'transformer_state': transformer_model.state_dict(),
                                'clip_model_state': model.state_dict(),
                                'optimizer_state': optimizer.state_dict(),
                                'scheduler_state': scheduler.state_dict(),
                                'best_angle': best_angle,
                            }, ckpt_path)
                            best_model_path = ckpt_path
                            save_msg = f"[CKPT] Saved improved epoch checkpoint to {ckpt_path}"
                            print(save_msg)
                            write_log(save_msg)
                        except Exception as e:
                            err_msg = f"[ERR] Failed to save epoch checkpoint: {e}"
                            print(err_msg)
                            write_log(err_msg)
        except Exception as _e_ep_eval:
            print(f"[EPOCH_EVAL][ERR] {str(_e_ep_eval)}")
    # 每轮训练后在训练集上推理并输出2D预测与GT
    if TRAIN_DATASET_NAME == "ETH-XGaze":
        transformer_model.eval()
        model.eval()
        pred_lines = []
        img_idx = 0
        pred_y_list = []
        label_y_list = []
        pred_norm_list = []
        label_norm_list = []
        with torch.no_grad():
            for batch in train_dl:
                raw_inputs = process_batch(batch, model=model)
                for k in raw_inputs:
                    if isinstance(raw_inputs[k], torch.Tensor):
                        raw_inputs[k] = raw_inputs[k].float()
                output = transformer_model(raw_inputs)
                output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                batch_size = output_safe.shape[0]
                for idx_in_batch in range(batch_size):
                    img_path = train_dl.dataset.inner_dataset.labels[img_idx].Face
                    def ccs_to_pitchyaw(vec):
                        x, y, z = vec
                        y = max(min(y, 1.0), -1.0)
                        pitch = math.asin(y)
                        yaw = math.atan2(x, z)
                        return pitch, yaw
                    pred_3d = output_safe[idx_in_batch].cpu().numpy()
                    label_3d = label_safe[idx_in_batch].cpu().numpy()
                    # 归一化
                    pred_3d_norm = np.linalg.norm(pred_3d) + 1e-6
                    label_3d_norm = np.linalg.norm(label_3d) + 1e-6
                    pred_3d_unit = pred_3d / pred_3d_norm
                    label_3d_unit = label_3d / label_3d_norm
                    pred_y_list.append(pred_3d[1])
                    label_y_list.append(label_3d[1])
                    pred_norm_list.append(pred_3d_norm)
                    label_norm_list.append(label_3d_norm)
                    # 只打印前20个样本的3D向量及y分量（未归一化和归一化后）
                    if img_idx < 20:
                        print(f"[DEBUG] img: {img_path}\npred_3d: {pred_3d}, pred_y: {pred_3d[1]:.6f}, pred_3d_unit: {pred_3d_unit}, pred_y_unit: {pred_3d_unit[1]:.6f}\nlabel_3d: {label_3d}, label_y: {label_3d[1]:.6f}, label_3d_unit: {label_3d_unit}, label_y_unit: {label_3d_unit[1]:.6f}")
                    pred_2d = ccs_to_pitchyaw(pred_3d_unit)
                    label_2d = ccs_to_pitchyaw(label_3d_unit)
                    pred_lines.append(f"{img_path},pred_pitch={pred_2d[0]:.6f},pred_yaw={pred_2d[1]:.6f},label_pitch={label_2d[0]:.6f},label_yaw={label_2d[1]:.6f}")
                    img_idx += 1
        # 统计归一化情况和y分量分布
        pred_y_arr = np.array(pred_y_list)
        label_y_arr = np.array(label_y_list)
        pred_norm_arr = np.array(pred_norm_list)
        label_norm_arr = np.array(label_norm_list)
        # print("[SUMMARY] pred_3d范数: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(pred_norm_arr.mean(), pred_norm_arr.std(), pred_norm_arr.min(), pred_norm_arr.max()))
        # print("[SUMMARY] label_3d范数: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(label_norm_arr.mean(), label_norm_arr.std(), label_norm_arr.min(), label_norm_arr.max()))
        # print("[SUMMARY] pred_y分布: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(pred_y_arr.mean(), pred_y_arr.std(), pred_y_arr.min(), pred_y_arr.max()))
        # print("[SUMMARY] label_y分布: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(label_y_arr.mean(), label_y_arr.std(), label_y_arr.min(), label_y_arr.max()))
        # 检查是否有y分量超出[-1,1]
        pred_y_out = np.sum((pred_y_arr < -1) | (pred_y_arr > 1))
        label_y_out = np.sum((label_y_arr < -1) | (label_y_arr > 1))
        # print(f"[SUMMARY] pred_y超出[-1,1]的数量: {pred_y_out}")
        # print(f"[SUMMARY] label_y超出[-1,1]的数量: {label_y_out}")
        pred_label_path = f"ethxgaze_train_pred_epoch{epoch}_with_gt.txt"
        with open(pred_label_path, "w", encoding="utf-8") as f:
            for line in pred_lines:
                f.write(line + "\n")
        print(f"训练集2D gaze及GT已保存到 {pred_label_path}")
    if TRAIN_DATASET_NAME == "ETH-XGaze":
        transformer_model.eval()
        model.eval()
        pred_lines = []
        img_idx = 0
        total_angle_sum = 0.0
        total_angle_n = 0
        with torch.no_grad():
            for batch in test_dl:
                raw_inputs = process_batch(batch, model=model)
                for k in raw_inputs:
                    if isinstance(raw_inputs[k], torch.Tensor):
                        raw_inputs[k] = raw_inputs[k].float()
                output = transformer_model(raw_inputs)
                output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                # compute per-sample angles and accumulate
                pred_n = output_safe / (output_safe.norm(dim=-1, keepdim=True) + 1e-6)
                gt_n = label_safe / (label_safe.norm(dim=-1, keepdim=True) + 1e-6)
                cos_sim = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
                angles = torch.acos(cos_sim) * 180.0 / math.pi
                total_angle_sum += angles.sum().item()
                total_angle_n += angles.numel()
                batch_size = output_safe.shape[0]
                for idx_in_batch in range(batch_size):
                    pred_3d = output_safe[idx_in_batch].cpu().numpy()
                    label_3d = label_safe[idx_in_batch].cpu().numpy()
                    # 归一化
                    pred_3d_unit = pred_3d / (np.linalg.norm(pred_3d) + 1e-6)
                    label_3d_unit = label_3d / (np.linalg.norm(label_3d) + 1e-6)
                    # 只输出pitch，空格分隔
                    def ccs_to_pitchyaw(vec):
                        x, y, z = vec
                        y = max(min(y, 1.0), -1.0)
                        pitch = math.asin(y)
                        return pitch
                    pred_pitch = ccs_to_pitchyaw(pred_3d_unit)
                    label_pitch = ccs_to_pitchyaw(label_3d_unit)
                    pred_lines.append(f"{pred_pitch:.6f} {label_pitch:.6f}")
                    img_idx += 1
        mean_angle_all = float('nan')
        if total_angle_n > 0:
            mean_angle_all = total_angle_sum / total_angle_n
        pred_label_path = f"ethxgaze_pred_pitch_epoch{epoch}_angle{mean_angle_all:.2f}_with_gt.txt"
        with open(pred_label_path, "w", encoding="utf-8") as f:
            for line in pred_lines:
                f.write(line + "\n")
        print(f"测试集pitch预测及GT已保存到 {pred_label_path}")
   

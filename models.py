# model


import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
import timm
from config import *
from torchvision.models.feature_extraction import create_feature_extractor

class SemanticPrototypeMixin:
    def _init_semantic_prototypes(
        self,
        illum_tokens,
        headpose_tokens,
        bg_tokens,
        desc_tokens,
    ):
        clip_model = copy.deepcopy(CLIP_MODEL)
        clip_model.eval()
        with torch.no_grad():
            illum_feats = clip_model.encode_text(illum_tokens).float()
            head_feats = clip_model.encode_text(headpose_tokens).float()
            bg_feats = clip_model.encode_text(bg_tokens).float()
            desc_feats = clip_model.encode_text(desc_tokens).float()
        del clip_model
        torch.cuda.empty_cache()

        self.illum_feats = nn.Parameter(illum_feats)
        self.head_feats = nn.Parameter(head_feats)
        self.bg_feats = nn.Parameter(bg_feats)
        self.desc_feats = nn.Parameter(desc_feats)
        self.semantic_norm_f1 = nn.LayerNorm(illum_feats.shape[-1])
        self.semantic_norm_f2 = nn.LayerNorm(illum_feats.shape[-1])

    def semantic_parameters(self):
        return [
            self.illum_feats,
            self.head_feats,
            self.bg_feats,
            self.desc_feats,
            *self.semantic_norm_f1.parameters(),
            *self.semantic_norm_f2.parameters(),
        ]

    def _select_prototype(self, img_norm, prototypes, scale):
        prototypes = prototypes.float()
        proto_norm = F.normalize(prototypes, dim=-1, eps=1e-6)
        scores = scale * img_norm @ proto_norm.T
        probs = F.softmax(scores, dim=-1)
        hard_idx = probs.argmax(dim=-1)
        hard = F.one_hot(hard_idx, num_classes=probs.size(-1)).type_as(probs)
        weights = hard - probs.detach() + probs
        selected = weights @ prototypes
        return selected, scores

    def compute_conditioned_features(self, img_feats):
        img_feats = img_feats.float()
        img_norm = F.normalize(img_feats, dim=-1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(max=50).float()

        selected_illum, sim_illum = self._select_prototype(
            img_norm, self.illum_feats, scale
        )
        selected_head, sim_head = self._select_prototype(
            img_norm, self.head_feats, scale
        )
        selected_bg, sim_bg = self._select_prototype(img_norm, self.bg_feats, scale)
        selected_desc, sim_desc = self._select_prototype(
            img_norm, self.desc_feats, scale
        )

        feature_1 = self.semantic_norm_f1(img_feats + selected_illum + selected_bg)
        feature_2 = self.semantic_norm_f2(img_feats + selected_desc + selected_head)
        aux = {
            "sim_illum": sim_illum,
            "sim_head": sim_head,
            "sim_bg": sim_bg,
            "sim_label": sim_desc,
            "img_norm": img_norm,
        }
        return feature_1, feature_2, aux

    def _flatten_or_pool_cnn_features(self, other_face, batch_size):
        feature_3 = self.main_model(other_face)
        if isinstance(feature_3, dict):
            feature_3 = feature_3["features"].mean(dim=(-2, -1))
        return feature_3.reshape(batch_size, -1)


class GEWithCLIPModel(SemanticPrototypeMixin, nn.Module):
    def __init__(
        self,
        irrelevant_feats_dim=512,
        relevant_feats_dim=512,
    ):
        super().__init__()

        self.illumination_texts = [
            "a face with bright light",
            "a face with low light",
            "a face with shadows",
        ]
        self.headpose_texts = [
            "a frontal face",
            "a profile face",
        ]
        self.background_texts = [
            "a face on bright background",
            "a face on dark background",
        ]
        self.label_texts = [
            "A photo of a face looking left",
            "A photo of a face looking upper left",
            "A photo of a face looking up",
            "A photo of a face looking upper right",
            "A photo of a face looking right",
            "A photo of a face looking lower right",
            "A photo of a face looking down",
            "A photo of a face looking lower left",
        ]
        illum_tokens = clip.tokenize(self.illumination_texts).to(DEVICE)
        headpose_tokens = clip.tokenize(self.headpose_texts).to(DEVICE)
        bg_tokens = clip.tokenize(self.background_texts).to(DEVICE)
        self.label_tokens = clip.tokenize(self.label_texts).to(DEVICE)
        self._init_semantic_prototypes(
            illum_tokens, headpose_tokens, bg_tokens, self.label_tokens
        )
        self.model = CLIP_MODEL
        self.encoder_i = CLIP_MODEL.encode_image
        self.encoder_t2 = CLIP_MODEL.encode_text

        if CNN_MODEL == "ResNet-50":
            # ResNet50 CNN for image features
            # 这个模型默认期望输入的数据是一个形状为 (batch_size, 3, 224(height), 224(width)) 的张量（Tensor）。
            main_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            main_model_feats_dim = main_model.fc.in_features
            main_model.fc = nn.Identity()
        elif CNN_MODEL == "ResNet-18":
            main_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            main_model_feats_dim = main_model.fc.in_features
            main_model.fc = nn.Identity()
        elif CNN_MODEL == "EdgeNeXt-Small":
            main_model = timm.create_model("edgenext_small", pretrained=True)
            main_model_feats_dim = main_model.head.fc.in_features
            main_model.head.fc = nn.Identity()
        self.main_model = main_model

        # fusion layers
        fused_dim = irrelevant_feats_dim + relevant_feats_dim + main_model_feats_dim
        self.fuse_model = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 3)  # 3D gaze output
        )
        self.logit_scale = CLIP_MODEL.logit_scale

    def forward(
        self,
        face,
        other_face,
    ):
        img_feats = self.encoder_i(face)
        feature_1, feature_2, aux = self.compute_conditioned_features(img_feats)
        sim_label = aux["sim_label"]
        feature_3 = self._flatten_or_pool_cnn_features(other_face, face.size(0))

        fused = torch.cat([feature_1, feature_2, feature_3], dim=-1)
        gaze_pred = self.fuse_model(fused)

        return gaze_pred, sim_label, feature_1, feature_2
class GEWithCLIPModel_zhao(SemanticPrototypeMixin, nn.Module):
    def __init__(
        self,
        irrelevant_feats_dim=512,
        relevant_feats_dim=512,
    ):
        super().__init__()

        self.illumination_texts = [
            "a face with bright light",
            "a face with low light",
            "a face with shadows",
        ]
        self.headpose_texts = [
            "a frontal face",
            "a profile face",
        ]
        self.background_texts = [
            "a face on bright background",
            "a face on dark background",
        ]
        self.label_texts = [
            "A photo of a face looking left",
            "A photo of a face looking upper left",
            "A photo of a face looking up",
            "A photo of a face looking upper right",
            "A photo of a face looking right",
            "A photo of a face looking lower right",
            "A photo of a face looking down",
            "A photo of a face looking lower left",
        ]
        illum_tokens = clip.tokenize(self.illumination_texts).to(DEVICE)
        headpose_tokens = clip.tokenize(self.headpose_texts).to(DEVICE)
        bg_tokens = clip.tokenize(self.background_texts).to(DEVICE)
        self.label_tokens = clip.tokenize(self.label_texts).to(DEVICE)
        self._init_semantic_prototypes(
            illum_tokens, headpose_tokens, bg_tokens, self.label_tokens
        )
        self.model = CLIP_MODEL
        self.encoder_i = CLIP_MODEL.encode_image
        self.encoder_t2 = CLIP_MODEL.encode_text

        if CNN_MODEL == "ResNet-50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # 使用 create_feature_extractor 提取 layer4 输出的特征图
            return_nodes = {"layer4": "features"}  # layer4 输出特征图
            main_model = create_feature_extractor(base_model, return_nodes=return_nodes)
            # main_model.forward(x) 返回一个字典，key 是 "features"，值形状为 [B, 2048, H, W]
            main_model_feats_dim = 2048  # ResNet50 layer4 的输出通道数

        elif CNN_MODEL == "ResNet-18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            return_nodes = {"layer4": "features"}
            main_model = create_feature_extractor(base_model, return_nodes=return_nodes)
            main_model_feats_dim = 512  # ResNet18 layer4 的输出通道数
        elif CNN_MODEL == "EdgeNeXt-Small":
            main_model = timm.create_model("edgenext_small", pretrained=True)
            main_model_feats_dim = main_model.head.fc.in_features
            main_model.head.fc = nn.Identity()
        self.main_model = main_model

        # fusion layers
        fused_dim = irrelevant_feats_dim + relevant_feats_dim + main_model_feats_dim
        self.fuse_model = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 3)  # 3D gaze output
        )
        self.logit_scale = CLIP_MODEL.logit_scale

    def forward(
        self,
        face,
        other_face,
    ):
        img_feats = self.encoder_i(face)
        feature_1, feature_2, aux = self.compute_conditioned_features(img_feats)
        sim_label = aux["sim_label"]
        feature_3 = self._flatten_or_pool_cnn_features(other_face, face.size(0))

        fused = torch.cat([feature_1, feature_2, feature_3], dim=-1)
        gaze_pred = self.fuse_model(fused)

        return gaze_pred, sim_label, feature_1, feature_2

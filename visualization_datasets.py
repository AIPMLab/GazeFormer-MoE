# 在使用 pyplot 前切换后端，避免 Qt 字体警告
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
# 全局字体设置：增大字体并使用 Times New Roman
plt.rcParams['font.size']   = 18
plt.rcParams['font.family'] = 'Times New Roman'
# 统一子图字体大小，与全局 font.size 保持一致
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})
# 公共画布大小
common_figsize = (7,6)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path  # added for custom Label path
# 只需要从 config 读取标签路径
from config import TRAIN_LABELS_PATH, TEST_LABELS_PATH
"""从 train.label 文件中读取第二、三、四列(X,Y,Z)，跳过首行字符串列"""
label_file = TRAIN_LABELS_PATH / "train.label"
# 解析 gaze pitch,yaw 并转换为 3D 单位向量
# 手动解析文件，逗号分隔，跳过非数值行，提取 X/Y/Z，并记录 pitch/yaw
coords_list = []
pitch_list = []
yaw_list = []
with open(label_file, 'r', encoding='utf-8') as f:
    next(f)  # 跳过 header
    for line in f:
        s = line.strip()
        if not s:
            continue
        # 提取第二字段（gaze）在首、次空格之间
        i1 = s.find(' ')
        if i1 < 0:
            continue
        i2 = s.find(' ', i1 + 1)
        if i2 < 0:
            continue
        gaze_str = s[i1+1:i2]
        try:
            pitch, yaw = map(float, gaze_str.split(','))
        except:
            continue
        # 由 pitch,yaw 转回 3D 向量：x = cos(pitch)*sin(yaw), y = sin(pitch), z = cos(pitch)*cos(yaw)
        x = np.cos(pitch) * np.sin(yaw)
        y = np.sin(pitch)
        z = np.cos(pitch) * np.cos(yaw)
        coords_list.append([x, y, z])
        pitch_list.append(pitch)
        yaw_list.append(yaw)
coords = np.array(coords_list)
print(f"[DEBUG] Loaded gaze vectors shape: {coords.shape}")
# 单位向量
unit_coords = coords
# 构建角度数组用于 2D 密度图
pitch_arr = np.array(pitch_list)
yaw_arr = np.array(yaw_list)


# 更科学的 pitch-yaw 二维密度图（Histogram2D + 对数归一化）
import matplotlib as mpl
# 转换为角度
pitch_deg = np.degrees(pitch_arr)
yaw_deg = np.degrees(yaw_arr)
fig2 = plt.figure(figsize=common_figsize)
ax2 = fig2.add_subplot(111)
hist2d = ax2.hist2d(
    yaw_deg, pitch_deg,
    bins=[720,360], range=[[-180,180],[-90,90]],  # increased for finer granularity
    cmap='viridis', norm=mpl.colors.LogNorm()
)

ax2.set_xlabel('Yaw (°)', fontname='Times New Roman')
ax2.set_ylabel('Pitch (°)', fontname='Times New Roman')
# 坐标轴刻度字体
for label in ax2.get_xticklabels(): label.set_fontname('Times New Roman')
for label in ax2.get_yticklabels(): label.set_fontname('Times New Roman')
# 添加 colorbar
cbar2 = fig2.colorbar(hist2d[3], ax=ax2, pad=0.02)
cbar2.set_label('Count (log scale)', fontname='Times New Roman')
for tick in cbar2.ax.get_yticklabels(): tick.set_fontname('Times New Roman')
ax2.set_aspect('auto')
# 自动裁剪到数据范围并添加小边距
min_x2, max_x2 = yaw_deg.min(), yaw_deg.max()
min_y2, max_y2 = pitch_deg.min(), pitch_deg.max()
margin_x2 = (max_x2 - min_x2) * 0.1
margin_y2 = (max_y2 - min_y2) * 0.1
ax2.set_xlim(min_x2 - margin_x2, max_x2 + margin_x2)
ax2.set_ylim(min_y2 - margin_y2, max_y2 + margin_y2)
# 细化刻度
xticks2 = np.linspace(min_x2 - margin_x2, max_x2 + margin_x2, 7)
yticks2 = np.linspace(min_y2 - margin_y2, max_y2 + margin_y2, 7)
ax2.set_xticks(xticks2)
ax2.set_yticks(yticks2)
for lbl in ax2.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax2.get_yticklabels(): lbl.set_fontname('Times New Roman')
plt.tight_layout()
plt.show()
  
# -----------------------------------------------------------------------------
# 处理另一数据集：从 config 中读取 TEST_LABELS_PATH
other_dir = Path(r"C:\Users\PS\Desktop\CLIP\datasets\EyeDiap\GazeHub\Label")
coords2 = []
for file in sorted(other_dir.iterdir()):
    if not file.is_file():
        continue
    with open(file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过 header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            vec_str = parts[4]
            try:
                x2, y2, z2 = map(float, vec_str.split(','))
            except:
                continue
            coords2.append([x2, y2, z2])
coords2 = np.array(coords2)
# 转换为数组并统计样本数
coords2 = np.array(coords2)
if coords2.size == 0:
    raise ValueError(f'No gaze vectors loaded from {other_dir}')
sample_count2 = coords2.shape[0]
print(f"[INFO] Other dataset loaded {sample_count2} samples")
# 计算角度 pitch/yaw（新数据集存储为视线方向向量，需取负值）
pitch2 = np.arcsin(-coords2[:, 1])
yaw2 = np.arctan2(-coords2[:, 0], -coords2[:, 2])
# 转为度
pitch2_deg = np.degrees(pitch2)
yaw2_deg = np.degrees(yaw2)

# 可视化另一数据集 Pitch-Yaw 二维密度图
fig3 = plt.figure(figsize=common_figsize)
ax3 = fig3.add_subplot(111)
hist2d_3 = ax3.hist2d(
    yaw2_deg, pitch2_deg,
    bins=[1440, 720], range=[[-180, 180], [-90, 90]],
    cmap='plasma', norm=mpl.colors.LogNorm()
)

ax3.set_xlabel('Yaw (°)', fontname='Times New Roman')
ax3.set_ylabel('Pitch (°)', fontname='Times New Roman')
# 坐标轴刻度字体
for lbl in ax3.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax3.get_yticklabels(): lbl.set_fontname('Times New Roman')
# 添加 colorbar
cbar3 = fig3.colorbar(hist2d_3[3], ax=ax3, pad=0.02)
cbar3.set_label('Count (log scale)', fontname='Times New Roman')
for tick in cbar3.ax.get_yticklabels(): tick.set_fontname('Times New Roman')

# 放大刻度范围，聚焦聚集区域
min_y = pitch2_deg.min(); max_y = pitch2_deg.max()
min_x = yaw2_deg.min(); max_x = yaw2_deg.max()
margin_x = (max_x - min_x) * 0.1
margin_y = (max_y - min_y) * 0.1
ax3.set_xlim(min_x - margin_x, max_x + margin_x)
ax3.set_ylim(min_y - margin_y, max_y + margin_y)
# 细化刻度
xticks = np.linspace(min_x - margin_x, max_x + margin_x, 7)
yticks = np.linspace(min_y - margin_y, max_y + margin_y, 7)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
# 设置刻度字体
for lbl in ax3.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax3.get_yticklabels(): lbl.set_fontname('Times New Roman')
ax3.set_aspect('auto')
# 自动裁剪到数据范围并添加小边距
min_x3, max_x3 = yaw2_deg.min(), yaw2_deg.max()
min_y3, max_y3 = pitch2_deg.min(), pitch2_deg.max()
margin_x3 = (max_x3 - min_x3) * 0.1
margin_y3 = (max_y3 - min_y3) * 0.1
ax3.set_xlim(min_x3 - margin_x3, max_x3 + margin_x3)
ax3.set_ylim(min_y3 - margin_y3, max_y3 + margin_y3)
# 细化刻度
xticks3 = np.linspace(min_x3 - margin_x3, max_x3 + margin_x3, 7)
yticks3 = np.linspace(min_y3 - margin_y3, max_y3 + margin_y3, 7)
ax3.set_xticks(xticks3)
ax3.set_yticks(yticks3)
for lbl in ax3.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax3.get_yticklabels(): lbl.set_fontname('Times New Roman')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 处理第三个数据集：请修改为实际 Label 文件夹路径
third_dir = Path(r"C:\Users\PS\Desktop\CLIP\datasets\Gaze360\GazeHub\Label")
coords3 = []
for file in sorted(third_dir.glob("*.label")):
    if not file.is_file():
        continue
    with open(file, 'r', encoding='utf-8') as f:
        next(f)  # 跳 header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            vec_str = parts[4]
            try:
                x3, y3, z3 = map(float, vec_str.split(','))
            except:
                continue
            coords3.append([x3, y3, z3])
coords3 = np.array(coords3)
sample_count3 = coords3.shape[0]

# 计算 pitch/yaw
pitch3 = np.arcsin(-coords3[:, 1])
yaw3 = np.arctan2(-coords3[:, 0], -coords3[:, 2])
# 转为度
pitch3_deg = np.degrees(pitch3)
yaw3_deg = np.degrees(yaw3)
# 2D 密度图
fig4 = plt.figure(figsize=common_figsize)
ax4 = fig4.add_subplot(111)
hist2d_4 = ax4.hist2d(
    yaw3_deg, pitch3_deg,
    bins=[720,360], range=[[-180,180],[-90,90]],
    cmap='magma', norm=mpl.colors.LogNorm()
)

ax4.set_xlabel('Yaw (°)', fontname='Times New Roman')
ax4.set_ylabel('Pitch (°)', fontname='Times New Roman')
for lbl in ax4.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax4.get_yticklabels(): lbl.set_fontname('Times New Roman')
cbar4 = fig4.colorbar(hist2d_4[3], ax=ax4, pad=0.02)
cbar4.set_label('Count (log scale)', fontname='Times New Roman')
for tick in cbar4.ax.get_yticklabels(): tick.set_fontname('Times New Roman')
ax4.set_aspect('auto')
# 自动裁剪到数据范围并添加小边距
min_x4, max_x4 = yaw3_deg.min(), yaw3_deg.max()
min_y4, max_y4 = pitch3_deg.min(), pitch3_deg.max()
margin_x4 = (max_x4 - min_x4) * 0.1
margin_y4 = (max_y4 - min_y4) * 0.1
ax4.set_xlim(min_x4 - margin_x4, max_x4 + margin_x4)
ax4.set_ylim(min_y4 - margin_y4, max_y4 + margin_y4)
# 细化刻度
xticks4 = np.linspace(min_x4 - margin_x4, max_x4 + margin_x4, 7)
yticks4 = np.linspace(min_y4 - margin_y4, max_y4 + margin_y4, 7)
ax4.set_xticks(xticks4)
ax4.set_yticks(yticks4)
for lbl in ax4.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax4.get_yticklabels(): lbl.set_fontname('Times New Roman')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 处理第六个数据集：Label 文件夹中多个 label 文件（第六列为 gaze 向量 x,y,z）
# 请根据实际路径修改 sixth_dir
sixth_dir = Path(r"C:\Users\PS\Desktop\CLIP\datasets\MPIIFaceGaze\GazeHub\Label")
coords6 = []
for file in sorted(sixth_dir.glob("*.label")):
    if not file.is_file():
        continue
    with open(file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过 header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            vec_str = parts[5]
            try:
                x6, y6, z6 = map(float, vec_str.split(','))
            except:
                continue
            coords6.append([x6, y6, z6])
coords6 = np.array(coords6)
sample_count6 = coords6.shape[0]
print(f"[INFO] Sixth dataset loaded {sample_count6} samples")
# 计算角度 pitch/yaw
pitch6 = np.arcsin(-coords6[:, 1])
yaw6 = np.arctan2(-coords6[:, 0], -coords6[:, 2])
# 转为度
pitch6_deg = np.degrees(pitch6)
yaw6_deg = np.degrees(yaw6)
# 2D 密度图
fig6 = plt.figure(figsize=common_figsize)
ax6 = fig6.add_subplot(111)
hist2d_6 = ax6.hist2d(
    yaw6_deg, pitch6_deg,
    bins=[1440,720], range=[[-180,180],[-90,90]],  # further increased bins for finer granularity
    cmap='cividis', norm=mpl.colors.LogNorm()
)

ax6.set_xlabel('Yaw (°)', fontname='Times New Roman')
ax6.set_ylabel('Pitch (°)', fontname='Times New Roman')
for lbl in ax6.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax6.get_yticklabels(): lbl.set_fontname('Times New Roman')
cbar6 = fig6.colorbar(hist2d_6[3], ax=ax6, pad=0.02)
cbar6.set_label('Count (log scale)', fontname='Times New Roman')
for tick in cbar6.ax.get_yticklabels(): tick.set_fontname('Times New Roman')
# 放大刻度范围，聚焦第六数据集聚集区域
min_y6 = pitch6_deg.min(); max_y6 = pitch6_deg.max()
min_x6 = yaw6_deg.min(); max_x6 = yaw6_deg.max()
margin_x6 = (max_x6 - min_x6) * 0.1
margin_y6 = (max_y6 - min_y6) * 0.1
ax6.set_xlim(min_x6 - margin_x6, max_x6 + margin_x6)
ax6.set_ylim(min_y6 - margin_y6, max_y6 + margin_y6)
# 细化刻度
xticks6 = np.linspace(min_x6 - margin_x6, max_x6 + margin_x6, 7)
yticks6 = np.linspace(min_y6 - margin_y6, max_y6 + margin_y6, 7)
ax6.set_xticks(xticks6)
ax6.set_yticks(yticks6)
for lbl in ax6.get_xticklabels(): lbl.set_fontname('Times New Roman')
for lbl in ax6.get_yticklabels(): lbl.set_fontname('Times New Roman')
ax6.set_aspect('auto')
plt.tight_layout()
plt.show()


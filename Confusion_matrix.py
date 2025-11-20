# Confusion_matrix.py
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_confusion_matrix(true_data,
                          pre_data,
                          path,
                          SNR,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    绘制并保存混淆矩阵。

    参数:
        true_data:  一维 list / numpy 数组，真实标签
        pre_data:   一维 list / numpy 数组，预测标签
        path:       保存目录（末尾不用加 /）
        SNR:        信噪比，用在文件名里
        normalize:  是否按行归一化（得到每类的召回率）
        title:      图标题（可为 None）
        cmap:       颜色映射
    """
    # ---------- 0. 基本检查 ----------
    if len(true_data) == 0 or len(pre_data) == 0:
        print("confusion_matrix: true_data 或 pre_data 为空，跳过绘图。")
        return

    true_data = np.asarray(true_data)
    pre_data = np.asarray(pre_data)

    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    # ---------- 1. 计算混淆矩阵 ----------
    cm = confusion_matrix(true_data, pre_data)
    num_classes = cm.shape[0]

    print("混淆矩阵 shape:", cm.shape)
    print("混淆矩阵:\n", cm)

    total_samples = cm.sum()
    diag_sum = np.trace(cm)
    overall_acc = diag_sum / total_samples if total_samples > 0 else 0.0
    print("Overall accuracy (from confusion matrix):", overall_acc)

    # 每一类的召回率（按行归一化）
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = cm.diagonal() / cm.sum(axis=1)
    # 某一类如果该行总数为 0，会出现 nan，这里忽略掉
    mean_diag = np.nanmean(per_class_recall)
    print("Mean diagonal value (mean recall per class):", mean_diag)

    # ---------- 2. 是否归一化显示 ----------
    if normalize:
        cm_display = per_class_recall.reshape(num_classes, 1) * np.eye(num_classes)
        # 更常规一点：直接按行归一化整张矩阵
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_norm, nan=0.0)
    else:
        cm_display = cm.astype(np.float32)

    # ---------- 3. 画图 ----------
    fig, ax = plt.subplots(figsize=(6, 6))
    if title is None:
        title = "Confusion matrix"

    ax.set_title(title, fontsize=16)

    # imshow 画矩阵
    if normalize:
        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)

    fig.colorbar(im, ax=ax)

    # 坐标刻度
    tick_marks = np.arange(num_classes)
    fontsize = 12 if num_classes <= 20 else 8
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(tick_marks, fontsize=fontsize, rotation=90)
    ax.set_yticklabels(tick_marks, fontsize=fontsize)

    ax.set_ylabel("True label", fontsize=18)
    ax.set_xlabel("Predicted label", fontsize=18)

    plt.tight_layout()

    # ---------- 4. 保存图像 ----------
    if normalize:
        fname = os.path.join(
            path,
            f"Confusion_matrix_{SNR}dB_acc{overall_acc:.4f}_meanDiag{mean_diag:.4f}_norm.svg",
        )
    else:
        fname = os.path.join(
            path,
            f"Confusion_matrix_{SNR}dB_acc{overall_acc:.4f}_raw.svg",
        )

    fig.savefig(fname)
    print("Confusion matrix saved to:", fname)
    plt.close(fig)

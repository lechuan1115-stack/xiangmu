#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import argparse
import time
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

import mydata_read
import mymodel1
import libmr
from collections import OrderedDict

# -----------------------
# 全局设置
# -----------------------
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser("Example")
parser.add_argument('--class_num', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=1)  # 推理场景通常为1
parser.add_argument('--lr-model', type=float, default=0.009, help="learning rate for model")
parser.add_argument('--model', type=str, default='P4AllCNN')  # CNN_Transformer_memory / CNN_Transformer
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--method', choices=['threshold','softmax','openmax'], default='openmax',
                    help="选择推理方法：threshold（阈值法），softmax（闭/混合集），openmax（OpenMax）")
args = parser.parse_args()

patience = 15
SNR = 20

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
if use_gpu:
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
else:
    print("####Currently using CPU")


# -----------------------
# 权重加载：自适配键名
# -----------------------
def smart_load_weights(model: nn.Module, ptpath: str):
    """
    自适配加载权重：
    - 去掉可能存在的 'module.' 前缀
    - 丢弃 BatchNorm 的 num_batches_tracked
    - strict=False，提高容错性（分类头或细微层名差异不阻塞）
    """
    state = torch.load(ptpath, map_location='cpu')
    clean_state = OrderedDict()
    for k, v in state.items():
        kk = k.replace('module.', '')
        if not kk.endswith('num_batches_tracked'):
            clean_state[kk] = v

    msd = model.state_dict()
    missing = [k for k in msd.keys() if k not in clean_state]
    unexpected = [k for k in clean_state.keys() if k not in msd]
    print(f'[load_info] will load {len(clean_state)}/{len(msd)} params, '
          f'missing={len(missing)}, unexpected={len(unexpected)}')

    model.load_state_dict(clean_state, strict=False)


# -----------------------
# 主流程
# -----------------------
def main():
    snr = 20
    unknown = 20

    indatapath = f'E:/项目所用代码/GCONV/G-CNN-main/luoyang_json/data/ADS-B_{snr}dB_test.mat'
    outdatapath = f'E:/项目所用代码/GCONV/G-CNN-main/luoyang_json/data/ADS-B_IQ_{unknown}test_{snr}dB.mat'
    ptpath = f'E:/项目所用代码/GCONV/G-CNN-main/luoyang_json/data/Memory_fixdata_20/P4AllCNN{snr}dB.pt'

    print("未知类数量：", unknown)
    udp_socket.sendto((" 未知类数量:" + str(unknown)).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("信噪比：", snr)
    udp_socket.sendto((" 信噪比:" + str(snr)).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    udp_socket.sendto((" ***************开始***************").encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    udp_socket.sendto((" 当前运行程序:OSR.py").encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("数据集为：", outdatapath)
    udp_socket.sendto((" 数据集为:" + outdatapath).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("模型为：", args.model)
    udp_socket.sendto((" 模型为:" + args.model).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("预训练模型：", ptpath)
    udp_socket.sendto((" 预训练模型:" + ptpath).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("超参数为：batch_size={}，lr_model={} earying_stop={}".format(args.batch_size, args.lr_model, patience))
    udp_socket.sendto((" 超参数为：batch_size={} lr_model={} earying_stop={}".format(
        args.batch_size, args.lr_model, patience)).encode('utf-8'), ('127.0.0.1', 40021))

    Pthreshold = 0.1
    udp_socket.sendto((" 阈值设置为:" + str(Pthreshold)).encode('utf-8'), ('127.0.0.1', 40021))


    # 选择方法并执行
    if args.method == 'threshold':
        threshold_osr(indatapath, outdatapath, ptpath, Pthreshold)
    elif args.method == 'softmax':
        softmax_osr(indatapath, outdatapath, ptpath)
    else:
        # OpenMax 需要用于拟合 MAV 的“训练集”路径（请按你的实际路径替换）
        mavdatapath = f'E:\项目所用代码\GCONV\G-CNN-main\luoyang_json\data\ADS-B_{snr}dB_train.mat'
        openmax_osr(mavdatapath, indatapath, outdatapath, ptpath, Pthreshold)
    # 如需 openmax，可改为：
    # mavdatapath = f'./ADS-B_100class/close_traindata/ADS-B_{snr}dB_train.mat'
    # openmax_osr(mavdatapath, indatapath, outdatapath, ptpath, Pthreshold)


# -----------------------
# Softmax（闭集+开集混合，仅示例）
# -----------------------
def softmax_osr(indatapath, outdatapath, ptpath):
    close_set = mydata_read.SignalDataset1(indatapath)
    open_set = mydata_read.SignalDataset1(outdatapath)
    mix_dataset = ConcatDataset([close_set, open_set])

    closeloader = DataLoader(close_set, batch_size=args.batch_size, shuffle=False)
    openloader = DataLoader(open_set, batch_size=args.batch_size, shuffle=False)
    mixloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=False)

    print("Creating model: {}".format(args.model))
    model = mymodel1.create(name=args.model, num_classes=args.class_num)
    model = model.to(device).eval()

    print("#######开始测试")
    smart_load_weights(model, ptpath)
    print('导入模型成功')

    start_time = time.time()
    real_label, pre_label, feature_save = test(model, mixloader)
    elapsed = time.time() - start_time
    print(len(mix_dataset))
    print("推理时间为:{:5f}ms".format(elapsed / len(mix_dataset) * 1000))


# -----------------------
# 通用测试（返回top-k等）
# -----------------------
def test(model, testloader):
    model.eval()
    real_label, pre_label, feature_save = [], [], []

    top1_count, top2_count, top3_count, top5_count, index = ([], [], [], [], 0)
    with torch.no_grad():
        for data, labels in testloader:
            data = data.float().to(device)

            # —— 统一把 labels 变成 [B] —— #
            if labels.dim() == 0:
                labels = labels.view(1)
            elif labels.dim() == 1:
                pass
            else:
                B = labels.size(0)
                labels = labels.view(B, -1)[:, 0]
            labels = labels.to(device).long()

            outputs, feature = model(data)             # [B, C], feature: [...]
            predictions = outputs.argmax(dim=1)        # [B]

            # 收集结果
            real_label.extend(labels.cpu().tolist())
            pre_label.extend(predictions.cpu().tolist())
            feature_save.append(feature.cpu().numpy().tolist())

            # 统计 top-k
            top_k = outputs.topk(5, 1, True, True).indices.cpu().tolist()
            B = outputs.size(0)
            for i in range(B):
                lab = int(labels[i].item())
                top1_count += [top_k[i][0] == lab]
                top2_count += [lab in top_k[i][0:2]]
                top3_count += [lab in top_k[i][0:3]]
                top5_count += [lab in top_k[i][0:5]]
                index += 1

    # 计算 top-k 准确率
    top1_acc = np.sum(top1_count) / index
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    top5_acc = float(np.sum(top5_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}% Top5:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100, top5_acc * 100
        )
    )
    return real_label, pre_label, feature_save



# -----------------------
# 阈值开集
# -----------------------
def threshold_osr(indatapath, outdatapath, ptpath, Pthreshold):
    close_set = mydata_read.SignalDataset1(indatapath)
    open_set = mydata_read.SignalDataset1(outdatapath)
    mix_dataset = ConcatDataset([close_set, open_set])
    mixloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=False)

    print("Creating model: {}".format(args.model))
    udp_socket.sendto((" Creating model:" + args.model).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)

    model = mymodel1.create(name=args.model, num_classes=args.class_num)
    model = model.to(device).eval()

    print("#######开始测试")
    udp_socket.sendto(("#######开始测试#######").encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)

    smart_load_weights(model, ptpath)
    print('导入模型成功')
    udp_socket.sendto(("#######导入模型成功#######").encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)

    start_time = time.time()

    real_label, pre_label = [], []
    known_classes = args.class_num
    unknown_idx = known_classes

    with torch.no_grad():
        for data, labels in mixloader:
            # data: [B, ...] ; labels: 可能是 [B], [B,1], [B,T] 等
            data = data.float().to(device)
            labels = labels.to(device)

            B = data.size(0)

            # 统一把 labels 变成 [B]：每样本只取第一个标量作为类别ID
            if labels.dim() == 0:
                lab = labels.view(1)
            elif labels.dim() == 1:
                lab = labels
            else:
                lab = labels.view(B, -1)[:, 0]   # 关键改动：只取每样本的第一个元素
            lab = lab.long()

            outputs, _ = model(data)             # [B, C]
            probs = F.softmax(outputs, dim=1)    # [B, C]
            max_prob, predlabel = probs.max(dim=1)  # [B], [B]
            pred_cls = predlabel.clone()

            # 低于阈值 → 未知类
            pred_cls[max_prob < Pthreshold] = unknown_idx

            # 标签 > C-1 → 未知类
            lab[lab > (known_classes - 1)] = unknown_idx

            # 追加到列表
            real_label.extend(lab.cpu().tolist())
            pre_label.extend(pred_cls.cpu().tolist())

    # 保险起见做一次长度一致性检查
    if len(real_label) != len(pre_label):
        print(f"[warn] y_true/y_pred length mismatch: {len(real_label)} vs {len(pre_label)}")
        # 尝试对齐
        n = min(len(real_label), len(pre_label))
        real_label = real_label[:n]
        pre_label = pre_label[:n]

    f1 = f1_score(real_label, pre_label, average='macro')
    p = precision_score(real_label, pre_label, average='macro')
    r = recall_score(real_label, pre_label, average='macro')

    print("macro-F1分数:", f1)
    udp_socket.sendto((" macro-F1分数:" + str(f1)).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("精确率macro-precision:", p)
    udp_socket.sendto((" 精确率macro-precision:" + str(p)).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("召回率macro-recall:", r)
    udp_socket.sendto((" 召回率macro-recall:" + str(r)).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)

    elapsed = time.time() - start_time
    print("测试样本数:", len(mix_dataset))
    udp_socket.sendto((" 测试样本数:" + str(len(mix_dataset))).encode('utf-8'), ('127.0.0.1', 40021))
    time.sleep(1)
    print("推理时间为:{:5f}ms".format(elapsed / len(mix_dataset) * 1000))
    udp_socket.sendto((" 推理时间为:" + str(elapsed / len(mix_dataset) * 1000) + "ms").encode('utf-8'),
                      ('127.0.0.1', 40021))
    time.sleep(1)


# -----------------------
# OpenMax 开集
# -----------------------
def openmax_osr(mavdatapath, indatapath, outdatapath, ptpath, Pthreshold):
    close_set = mydata_read.SignalDataset1(indatapath)
    open_set = mydata_read.SignalDataset1(outdatapath)
    mav_set = mydata_read.SignalDataset1(mavdatapath)

    mix_dataset = ConcatDataset([close_set, open_set])

    mavloader = DataLoader(mav_set, batch_size=args.batch_size, shuffle=False)
    mixloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=False)

    print("Creating model: {}".format(args.model))
    model = mymodel1.create(name=args.model, num_classes=args.class_num)
    model = model.to(device).eval()

    print("#######开始测试")
    smart_load_weights(model, ptpath)

    MAV, WeibullParams = OpenMax(model, mavloader, device)
    udp_socket.sendto((" weibull拟合完成").encode('utf-8'), ('127.0.0.1', 40021))

    openmaxtest(mixloader, model, device, MAV, WeibullParams, float(Pthreshold))


def OpenMax(model, loader, device):
    model.eval()

    known_classes = args.class_num  # 100
    MAV = torch.zeros(known_classes, known_classes, device=device)  # [C, C]
    AVNUM = torch.zeros(known_classes, 1, device=device)           # [C, 1]

    with torch.no_grad():
        for x, y in loader:
            data = x.to(device).float()
            labels = y.to(device).squeeze()  # scalar for bs=1
            out, _ = model(data)             # out: [1, C]
            AV = out.squeeze(0)              # [C]
            pred = int(out.argmax(dim=1).item())
            lab = int(labels.item())

            # 只有预测正确才参与 MAV
            if pred == lab:
                MAV[pred, :] = MAV[pred, :] + AV
                AVNUM[pred] = AVNUM[pred] + 1

    # 避免除零
    AVNUM_safe = AVNUM.clone()
    AVNUM_safe[AVNUM_safe == 0] = 1
    MAV = MAV / AVNUM_safe

    # 统计 AV 到 MAV 的距离
    max_per_class = 400  # 每类样本数上限（根据你数据集情况可调）
    DIST = -1 * torch.ones(known_classes, max_per_class, device=device)
    NUM = torch.zeros(known_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for x, y in loader:
            data = x.to(device).float()
            labels = y.to(device).squeeze()
            out, _ = model(data)
            AV = out.squeeze(0)
            pred = int(out.argmax(dim=1).item())
            lab = int(labels.item())
            if pred == lab:
                idx = int(NUM[pred].item())
                if idx < max_per_class:
                    DIST[pred, idx] = torch.dist(AV, MAV[pred, :], p=2)
                    NUM[pred] = NUM[pred] + 1

    # 韦伯拟合
    WeibullParams = []
    tailSize = 20
    for i in range(known_classes):
        mr = libmr.MR()
        # 过滤未填充位置
        valid = DIST[i][DIST[i] >= 0]
        if valid.numel() == 0:
            # 没有有效样本，放占位参数
            WeibullParams.append((0, 0, 0, 0, 0))
            continue
        sortDIST = torch.sort(valid, descending=True).values.cpu().numpy()
        fit_tail = min(tailSize, len(sortDIST))
        mr.fit_high(sortDIST, fit_tail)
        WeibullParams.append(mr.get_params())

    udp_socket.sendto((" ********************结束********************").encode('utf-8'), ('127.0.0.1', 40021))
    return MAV, WeibullParams


def openmaxtest(loader, model, device, MAV, WeibullParams, Pthreshold):
    """
    OpenMax 推理（兼容 Python 3.6/3.7，无海象运算符）
    - 将标签统一为 [B] 形状
    - 逐样本计算到各类 MAV 的距离、Weibull 概率、修正分数与未知类分数
    - 开集判决：若 prob < 阈值 或 未知类胜出 → 预测为未知类 (编号 = args.class_num)
    """
    model.eval()
    start_time = time.time()

    real_label, pre_label = [], []
    C = args.class_num               # 已知类数，例如 100
    unknown_idx = C                  # 未知类编号 = C

    with torch.no_grad():
        for x, y in loader:
            data = x.to(device).float()
            labels = y.to(device)

            # —— 统一把 labels 标准化为 [B] —— #
            if labels.dim() == 0:
                labels = labels.view(1)
            elif labels.dim() == 1:
                pass
            else:
                B_tmp = labels.size(0)
                labels = labels.view(B_tmp, -1)[:, 0]
            labels = labels.long()                 # [B]
            B = data.size(0)

            out, _ = model(data)                   # [B, C]

            for i in range(B):
                AV = out[i]                        # [C]，该样本的激活向量

                # 计算该样本 AV 到每个 MAV 的距离
                eachdist = torch.zeros(C, device=device)
                for j in range(C):
                    eachdist[j] = torch.dist(AV, MAV[j, :], p=2)

                # 按距离升序排序（越小越相近）
                eachdst = eachdist
                order = torch.argsort(eachdst, descending=False)

                # 计算 w（不属于第 i 类的概率）
                w = torch.zeros(C, device=device)
                for rank in range(order.numel()):
                    k = int(order[rank].item())
                    mr = libmr.MR()
                    a, b, c, d, e = WeibullParams[k]
                    mr.set_params(a, b, c, d, e)
                    # alpha 取 C（类数）
                    dist_k = float(eachdst[k].item())
                    w[k] = 1.0 - (float(C) - float(rank)) / float(C) * (1.0 - mr.cdf(dist_k))

                # 矫正分数：已知类分数 & 未知类分数
                modAV = AV * (1.0 - w)                              # [C]
                score_unknown = torch.sum(AV * w) / float(C)        # 标量

                all_score = torch.cat((modAV.view(1, -1), score_unknown.view(1, -1)), dim=1)  # [1, C+1]
                PROB = F.softmax(all_score, dim=-1)
                prob = float(PROB.max().item())

                pred_known = int(out[i].argmax(dim=0).item())       # 已知类里得分最高者
                pred_all = int(all_score[0].argmax(dim=0).item())   # C+1 类中的最大（最后一类是未知）

                # 开集判决：概率低或“未知类”胜出 → 归为未知
                if prob < Pthreshold or pred_all == C:
                    pred_cls = unknown_idx
                else:
                    pred_cls = pred_known

                # 规范化真实标签（>C-1 视为未知）
                lab_i = int(labels[i].item())
                if lab_i > (C - 1):
                    lab_i = unknown_idx

                real_label.append(lab_i)
                pre_label.append(pred_cls)

    # 评估
    real_label = np.asarray(real_label)
    pre_label = np.asarray(pre_label)
    f1 = f1_score(real_label, pre_label, average='macro')
    p = precision_score(real_label, pre_label, average='macro')
    r = recall_score(real_label, pre_label, average='macro')

    print("macro-F1分数:", f1)
    udp_socket.sendto((" macro-F1分数:" + str(f1)).encode('utf-8'), ('127.0.0.1', 40021))
    print("精确率macro-precision:", p)
    udp_socket.sendto((" 精确率macro-precision:" + str(p)).encode('utf-8'), ('127.0.0.1', 40021))
    print("召回率macro-recall:", r)
    udp_socket.sendto((" 召回率macro-recall:" + str(r)).encode('utf-8'), ('127.0.0.1', 40021))

    elapsed = time.time() - start_time
    # 注意：len(loader) 是 batch 数；若要每样本时间，用 len(loader.dataset)
    print("测试batch数:", len(loader))
    udp_socket.sendto((" 测试batch数:" + str(len(loader))).encode('utf-8'), ('127.0.0.1', 40021))
    print("推理时间为:{:5f}ms/batch".format(elapsed / len(loader) * 1000))
    udp_socket.sendto((" 推理时间为:" + str(elapsed / len(loader) * 1000) + "ms/batch").encode('utf-8'),
                      ('127.0.0.1', 40021))




# -----------------------
# 辅助
# -----------------------
def softmax(x):
    return F.softmax(x, dim=-1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch.nn.functional as F
import os
import sys
import argparse
import datetime
import time
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from pytorchtools import EarlyStopping
import mydata_read
import mymodel1
from utils import AverageMeter, Logger
from torch.utils.data import DataLoader,ConcatDataset
from warmup_scheduler import GradualWarmupScheduler
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser("Example")

#参数配置
# dataset：数据集设置
parser.add_argument('--class_num', type=int, default=10)#class_num表示类别数量（分类任务的类数），默认为100类
parser.add_argument('-j', '--workers', default=0, type=int,#-j/-workers表示数据加载的进程，“0”表示不使用多线程
                     help="number of data loading workers (default: 4)")
parser.add_argument('--batch-size', type=int, default=64)#batch-size批量大小（每次输入模型的数据量）是256
parser.add_argument('--lr-model', type=float, default=0.009, help="learning rate for model")# lr-model 模型的学习率，对于wifi数据集开始的学习率应该设置大点
parser.add_argument('--lr-cent', type=float, default=0.1, help="learning rate for center loss")#中心损失的学习率
parser.add_argument('--max-epoch', type=int, default=100)  #训练总轮次
parser.add_argument('--stepsize', type=int, default=10) # 学习率的调整步长：每隔多少epoch调整一次学习率
parser.add_argument('--gamma', type=float, default=0.75, help="learning rate decay")  # 学习率衰减因子，每stepsize轮，学习率会乘以这个因子

# model：模型结构
parser.add_argument('--model', type=str, default='P4AllCNN') #选择使用的模型结构：P4AllCNN   CNN_Transformer_memory  CNN_Transformer

# misc：杂项设置
parser.add_argument('--eval-freq', type=int, default=1)# 模型验证评估的频率，每多少个epoch在验证集上评估一次模型，默认为每轮都评估
parser.add_argument('--print-freq', type=int, default=50)#打印训练状态的频率，每50batch打印一次
parser.add_argument('--gpu', type=str, default='0')#指定使用的GPU编号，例如“0”表示第一块显卡
parser.add_argument('--seed', type=int, default=42)#设置随机种子（随机数生成器的初始化器），无论运行多少次，结果都会一样，确保训练过程可复现
parser.add_argument('--use-cpu', action='store_true')#如果命令行中输入了-use-cpu，则使用cpu，不输入则默认为False，用GPU
# store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False。反之亦然，store_false也就是默认为True，一旦命令中有此参数，其值则变为False。
parser.add_argument('--save-dir', type=str, default='Memory_fixdata_20/')#数据存储的位置  Memory Transformer
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")#输入了-plot参数，则在每轮训练后绘制特征图用于可视化，默认不绘图

#参数解析
args = parser.parse_args()#将上面定义的所有参数从命令行读取并解析，最终得到args变量在程序中调用

#数据路径设置与文件名拼接
file ='./data/'# file ='/media/liuchang/Expansion1/ADS-B_luoyang/'

def main():
    # SNR = 4#
    # filepath = file + 'iq_DATAIQ2_ch1_40M_100x100_LOS_4096_{}dB_FADE_5A.mat'.format(SNR)
    for SNR in range(20,22,2):
        filepath = file + 'ADS-B_{}dB_train.mat'.format(SNR)#遍历SNR=20到22，步长为2，拼接构建出数据路径文件
        # trainfilepath = file + 'close_train_20/ADS-B_{}dB_train.mat'.format(SNR)
        # valfilepath = file + 'close_validation_20/ADS-B_{}dB_validation.mat'.format(SNR)
        # testfilepath = file + 'close_test_20/ADS-B_{}dB_test.mat'.format(SNR)
        # print(filepath)
        # 根据SNR大小设置模型保存路径和日志输出文件，模型.pt,日志.txt
        if SNR < 50:
            path1 = osp.join(file + args.save_dir + args.model + str(SNR) + "dB.pt")
            sys.stdout = Logger(osp.join(file + args.save_dir, args.model + '_log_' + str(SNR) + 'dB.txt'))
        else:
            path1 = osp.join(file + args.save_dir + args.model + ".pt")
            sys.stdout = Logger(osp.join(file + args.save_dir, args.model + '_log.txt'))
        # 设置早停机制的耐心值，二分类任务容许更多轮训练，耐心值较大
        if args.class_num == 2:#二分类
            patience = 25
        else:
            patience = 15
        # 打印当前文件，模型结构，训练超参数...
        print("当前运行程序：5G信号仿真.py")
        print("数据集为：", filepath)
        print("模型为：", args.model)
        print("超参数为：batch_size={}，lr_model={} earying_stop={}".format(args.batch_size, args.lr_model, patience))
        #环境设置（是否使用GPU）
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        use_gpu = torch.cuda.is_available()
        if args.use_cpu: use_gpu = False
        if use_gpu:
            cudnn.benchmark = True
            torch.manual_seed(args.seed)
        else:
            print("####Currently using CPU")
        #数据读取模块：根据模型类型选择不同的数据读取类
        if args.model == 'P4AllCNN':
            print("导入kuozhan1维数据")#使用群等变卷积需要扩展一维数据
            data_set = mydata_read.SignalDataset1(filepath)
            # validate_set = mydata_read.SignalDataset1(valfilepath)
            # test_set = mydata_read.SignalDataset1(testfilepath)
        else:
            train_set = mydata_read.SignalDataset2(filepath)
            # validate_set = mydata_read.SignalDataset2(valfilepath)
            # test_set = mydata_read.SignalDataset2(testfilepath)
            # mix_dataset = ConcatDataset([train_set, validate_set])

        #数据划分：8:1:1切分成训练，验证，测试
        length = len(data_set)
        # print(length)
        train_size, validate_size, test_size = int(0.8 * length), int(0.1 * length), length - int(0.8 * length) - int(
            0.1 * length)
        validate_set, train_set, test_set = torch.utils.data.random_split(data_set,
                                                                   [validate_size, train_size, test_size])
        # 加载器构建，按批次（batch）将 Dataset 中的数据高效加载进模型训练流程中的工具。使用shuffle=True保证每轮训练数据顺序不同
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(validate_set, batch_size=args.batch_size, shuffle=True)
        # shuffle=True用于打乱数据集，每次都会以不同的顺序返回  每一个epoch都要输入trainloader 每次循环trainloader的内容都不同
        testloder = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
        print("Creating model: {}".format(args.model))

        #模型构建与设备转移：动态创建模型，并转移到GPU
        model = mymodel1.create(name=args.model, num_classes=args.class_num)
        # model.apply(weight_init_1)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # if use_gpu:
        #     model = nn.DataParallel(model).cuda()
        #     model = model.cuda()#该注释部分多GPU时或者训练不稳定时适合使用
        model = model.cuda()#将模型搬入GPU内存，以加速训练
        criterion_xent = nn.NLLLoss() # 定义交叉熵损失函数

        # 定义CNN的优化器
        # optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model)#adam优化器
        optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)#sgd优化器，weight_decay：权重衰减，momentum：动量，用于加速收敛速度
        if args.stepsize > 0:
            scheduler1 = lr_scheduler.ExponentialLR(optimizer_model, 0.9, last_epoch=-1)#使用指数衰减调度器，每次调用scheduler.step()，学习率乘以 0.9。
            # scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)# 适用于一开始学习率小的网络
            scheduler_warmup = GradualWarmupScheduler(optimizer_model, multiplier=1, total_epoch=8, after_scheduler=scheduler1)#使用warmup（预热）调度器，再接scheduler1。前 8 个 epoch 用较慢的线性学习率上升（warmup），第9个 epoch 接入指数衰减调度器 scheduler1，有利于稳定训练初期。
        # print(path1)
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=path1)#早停机制：启用 EarlyStopping，当验证损失连续 patience 个 epoch 没有改善时，提前停止训练，并保存最佳模型到 path1。
        # 初始化各类统计信息，分别记录训练验证的准确率与损失
        acc_train = []
        acc_eval = []
        loss_train = []
        loss_eval = []
        start_time = time.time()#记录训练耗时
        # for epoch in range(args.max_epoch):#开始训练主循环，迭代最大轮数，每一轮相当于用全部数据训练一遍
        #     print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))  # {}  打印当前轮次
        #     acctrain, losstrain = train(model, criterion_xent, optimizer_model, trainloader, use_gpu, args.class_num,epoch)#用当前训练集trainloader进行训练，返回该轮的训练的准确率和损失
        #     loss_train.append(losstrain)
        #     acc_train.append(acctrain)#保存当前轮的训练记录
        #     if args.stepsize > 0:  # and epoch < 61
        #         scheduler_warmup.step()#在每个epoch后更新学习率
        #         # scheduler1.step()
        #     if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:#满足“当前 epoch 为 eval 频率的倍数”或“已到最后一轮”，就做验证。
        #         acc, losseval = eval(model, criterion_xent, valloader, use_gpu)#用验证集valloader评估模型性能，返回准确率和损失
        #         print("Train_Accuracy (%): {:.4f} \t Eval_Accuracy (%): {:.4f}  ".format(acctrain, acc))#打印训练验证准确率
        #         print("Train_Loss (%): {:.3f} \t Eval_Loss (%): {:.3f}  ".format(losstrain, losseval))#打印训练验证损失
        #         acc_eval.append(acc)
        #         loss_eval.append(losseval)#保存验证记录
        #         early_stopping(losseval, model)#调用早停机制，如果验证损失没有下降，就累计一次“早停触发记录”
        #         if early_stopping.early_stop:
        #             print("Early stopping")
        #             break#验证是否早停，如果连续patience轮都没提升，停止训练
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("训练时间Finished. Total elapsed time (h:m:s): {}".format(elapsed))#计算并打印训练总耗时时间
        #绘制准确率曲线
        # fig = plt.figure(1,dpi=115)
        # np.save(osp.join(file + args.save_dir, "acc_train.npy"), acc_train)
        # np.save(osp.join(file + args.save_dir, "acc_eval.npy"), acc_eval)
        # plt.plot(acc_train, 'o-',linewidth=1.4,label="train_acc")
        # plt.plot(acc_eval,'^-',linewidth=1.4, label="eval_acc")
        # plt.ylabel('Accuracy', fontsize=25, labelpad=9.5)
        # plt.xlabel('number of iterations', fontsize=25, labelpad=9.5)
        # plt.rcParams.update({'font.size': 15})
        # plt.legend()
        # plt.xticks(fontsize=18)
        # plt.yticks([0,10,20,30,40,50,60,70,80,90,100],fontsize=18)
        # plt.show()
        # fig.savefig(osp.join(file + args.save_dir,str(SNR)+ "dB_acc.svg"))

        #绘制损失曲线
        # fig = plt.figure(2,dpi=115)
        # np.save(osp.join(file + args.save_dir, "loss_train.npy"), loss_train)
        # np.save(osp.join(file + args.save_dir, "loss_eval.npy"), loss_eval)
        # plt.plot(loss_train, 'o-', linewidth=1.4,label="train_loss")
        # plt.plot(loss_eval, '^-',linewidth=1.4,label="eval_loss")
        # plt.ylabel('Loss', fontsize=25, labelpad=9.5)
        # plt.xlabel('number of iterations', fontsize=25, labelpad=9.5)
        # plt.rcParams.update({'font.size': 15})
        # plt.legend()
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.show()
        # fig.savefig(osp.join(file + args.save_dir, str(SNR)+ "dB_loss.svg"))

        print("#######开始测试")
        model.load_state_dict(torch.load(path1, map_location='cuda' if use_gpu else 'cpu'))
        print('导入模型成功')

        start_time = time.time()
        real_label, pre_label, feature_save = test(model, testloder, use_gpu)
        elapsed = time.time() - start_time
        print(len(test_set))
        print("推理时间为:{:5f}ms".format(elapsed / len(test_set) * 1000))

        # ------- 展平标签 -------
        reallabel, predlabel = [], []
        for i in range(len(real_label)):
            reallabel += real_label[i]
            predlabel += pre_label[i]

        reallabel = list(map(int, reallabel))
        predlabel = list(map(int, predlabel))

        print("真实标签样本数:", len(reallabel))
        print("预测标签样本数:", len(predlabel))
        print("前20个真实标签:", reallabel[:20])
        print("前20个预测标签:", predlabel[:20])

        # ------- 计算整体样本级准确率（和 Top1 应该一致） -------
        if len(reallabel) > 0:
            overall_acc = np.mean(np.array(reallabel) == np.array(predlabel))
        else:
            overall_acc = 0.0
        print("Overall accuracy (sample-wise): {:.4f}".format(overall_acc))

        # ------- 保存特征 -------
        feature_temp = [torch.tensor(f) for f in feature_save]
        if len(feature_temp) > 0:
            feature = torch.cat(feature_temp, dim=0)
            np.save(
                osp.join(file + args.save_dir, f"{SNR}feature.npy"),
                feature.cpu().numpy()
            )
        else:
            print("feature_save 为空，没有特征需要保存。")

        # ------- 画混淆矩阵 -------
        import Confusion_matrix
        Confusion_matrix.plot_confusion_matrix(
            reallabel,
            predlabel,
            file + args.save_dir,
            SNR,
            normalize=True,
            cmap=plt.cm.Blues
        )
        plt.close()


#训练阶段
def train(model, criterion_xent, optimizer_model, trainloader,use_gpu, num_classes, epoch):
# 定义训练函数，输入包括模型，交叉熵损失函数，优化器，训练数据加载器，是否使用GPU，类别数，当前轮数
    model.train()#将模型设置为训练模式
    losses = AverageMeter()#初始化一个记录平均损失的工具，用于统计损失的平均值
    # metric_criterion = MetricLoss()
    correct, total = 0, 0#初始化分类正确数和总样本数，用于后续计算准确率
    # running_metric_loss = 0.0
    for batch_idx, (data, labels) in enumerate(trainloader):#从训练数据加载器中逐批获取数据及标签
        data = data.float()#将数据转换为float类型，确保后续输入网络兼容
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()# .type(torch.complex64)。如果使用GPU，加速模型训练，将数据与标签拷贝到GPU上
        outputs, feature = model(data)#将数据输入模型，得到分类输出和中间特征

        # class_out = feature
        # feats = np.array(class_out)  # t-SNE可视化的特征输入必须是np.array格式
        # # print(feats.shape)
        # joblib.dump(feats, 'feat_CNN1.pkl')#保存特征
        # outputs, aux = model(data)

        preds = F.log_softmax(outputs, dim=1)#对输出使用 log softmax（常用于与 NLLLoss 配合使用），维度1是类别维
        labels = torch.squeeze(labels)#去除标签中的多余维度
        # metric_loss = metric_criterion(feature, labels.long())
        loss_xent = criterion_xent(preds, labels.long())  # 计算交叉熵损失函数：计算输出和标签的损失
        loss = loss_xent  # loss在更新时会改变参数，使loss_cent和loss_xent同时变化
        # loss += 0.3 * metric_loss
        optimizer_model.zero_grad()#梯度清零，避免上次反向传播的梯度累积
        # optimizer_centloss.zero_grad()
        loss.backward()#反向传播，计算梯度
        optimizer_model.step()#用优化器更新模型参数
        losses.update(loss.item(), labels.size(0))#更新平均损失
        # running_metric_loss += metric_loss.item()
        # running_metric_loss = running_metric_loss / len(trainloader)
        total += len(preds)
        correct += (preds.argmax(dim=1) == labels).sum().item()#计算本批次预测正确数量，并更新数量

    acc = correct * 100. / total
    return acc, losses.avg#计算总准确率并返回平均损失

#验证阶段
def eval(model, criterion_xent, testloader, use_gpu):#定义评估函数，输入为模型...
    model.eval()#切换为评估模式
    losses = AverageMeter()#初始化平均损失
    # metric_criterion = MetricLoss()
    correct, total = 0, 0 # .type(torch.complex64)。初始化准确率统计变量
    with torch.no_grad():#在评估阶段关闭梯度计算，加快计算速度，节省内存
        for data, labels in testloader:#按批历测试数据
            data = data.float()#预处理数据
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()#必要时放到GPU
            labels = torch.squeeze(labels)
            outputs, feature = model(data)#预测结果及特征提取

            # outputs, aux = model(data)
            preds = F.log_softmax(outputs, dim=1)#对输出应用log_softmax
            # metric_loss = metric_criterion(feature, labels.long())
            eval_loss = criterion_xent(preds, labels.long())#计算交叉熵损失
            # eval_loss += 0.3 * metric_loss
            losses.update(eval_loss.item(), labels.size(0))#更新平均损失
            total += len(preds)
            correct += (preds.argmax(dim=1) == labels).sum().item()#预测统计个数
    acc = correct * 100. / total
    # err = 100. - acc
    return acc, losses.avg#返回准确率与平均损失


def test(model, testloader, use_gpu):
    """
    测试阶段：
    - 逐 batch 推理
    - 统计 Top1 / Top2 / Top3 / Top5 准确率
    - 返回:
        real_label   : List[List[int]]   每个 batch 的真实标签
        pre_label    : List[List[int]]   每个 batch 的预测标签
        feature_save : List[np.ndarray]  每个 batch 的特征 (batch, feat_dim)
    """
    model.eval()
    real_label, pre_label, feature_save = [], [], []

    top1_count, top2_count, top3_count, top5_count = [], [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            data = data.float()
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            labels = torch.squeeze(labels)

            # 前向推理
            outputs, feature = model(data)

            # ------- 保存真实标签 / 预测标签 / 特征 -------
            preds = outputs.argmax(dim=1)  # (batch,)
            real_label.append(labels.cpu().numpy().tolist())   # List[int]
            pre_label.append(preds.cpu().numpy().tolist())     # List[int]
            feature_save.append(feature.cpu().numpy())         # (batch, feat_dim)

            # ------- 计算 top-k -------
            _, top_k = outputs.topk(5, 1, True, True)          # (batch, 5)
            topk_result = top_k.cpu().tolist()

            for i in range(labels.shape[0]):
                gt = int(labels[i].item())
                top1_count.append(topk_result[i][0] == gt)
                top2_count.append(gt in topk_result[i][0:2])
                top3_count.append(gt in topk_result[i][0:3])
                top5_count.append(gt in topk_result[i][0:5])

    # ------- 计算各个 top-k 准确率 -------
    top1_acc = float(np.mean(top1_count)) if len(top1_count) > 0 else 0.0
    top2_acc = float(np.mean(top2_count)) if len(top2_count) > 0 else 0.0
    top3_acc = float(np.mean(top3_count)) if len(top3_count) > 0 else 0.0
    top5_acc = float(np.mean(top5_count)) if len(top5_count) > 0 else 0.0

    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}% Top5:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100, top5_acc * 100
        )
    )

    return real_label, pre_label, feature_save



# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight)
#         # nn.init.constant_(m.bias, 0)
#     # 也可以判断是否为conv2d，使用相应的初始化方式
#     elif isinstance(m, nn.Conv1d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#      # 是否为批归一化层
#     elif isinstance(m, nn.BatchNorm1d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
#         nn.init.orthogonal(m.weight_ih_l0)
#         nn.init.orthogonal(m.weight_hh_l0)
#         nn.init.constant_(m.bias_ih_l0, 0)
#         nn.init.constant_(m.bias_hh_l0, 0)
#
#
#
# def weight_init_RNN(m):# 主要用以解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法。
#     if isinstance(m, (nn.Conv1d, nn.Linear)):
#         nn.init.orthogonal(m.weight)
#
# def weight_init_1(m):
#     if isinstance(m, (nn.Conv1d)):
#         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#     elif isinstance(m, (nn.Linear)):
#         nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    main()


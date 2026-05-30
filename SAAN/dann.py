import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split
import os

import DaNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
MOMENTUM = 0.9
L2_WEIGHT = 5e-4
N_EPOCH = 50
BATCH_SIZE = [32, 32]

# 指标存储
RESULT_SOURCE_TRAIN = []
RESULT_TARGET_TRAIN = []
RESULT_TARGET_VAL = []

log_train = open('log_DANN_ViT_train.txt', 'w')
log_val = open('log_DANN_ViT_val.txt', 'w')
log_train.write("Epoch,Source Training Loss,Source Training Acc\n")
log_val.write("Epoch,Target Training Acc,Target Validation Acc\n")


def train(model, optimizer, epoch, data_src, data_tar_train):
    model.train()
    total_loss = 0.0
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    correct_src = 0

    tar_iter = iter(data_tar_train)
    alpha = 1.0

    for data, target in data_src:
        try:
            x_tar, _ = next(tar_iter)
        except StopIteration:
            tar_iter = iter(data_tar_train)
            x_tar, _ = next(tar_iter)

        bs = min(data.size(0), x_tar.size(0))
        data, target = data[:bs], target[:bs]
        x_tar = x_tar[:bs]

        data = data.to(DEVICE)
        target = target.to(DEVICE)
        x_tar = x_tar.to(DEVICE)

        src_label, src_domain = model(data, alpha)
        _, tar_domain = model(x_tar, alpha)

        domain_src = torch.zeros(bs).long().to(DEVICE)
        domain_tar = torch.ones(bs).long().to(DEVICE)

        loss_cls = criterion_cls(src_label, target)
        loss_domain = criterion_domain(src_domain, domain_src) + criterion_domain(tar_domain, domain_tar)
        loss = loss_cls + loss_domain

        pred = src_label.argmax(1)
        correct_src += pred.eq(target).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(data_src)
    src_acc = 100.0 * correct_src / len(data_src.dataset)

    res = f'Epoch: [{epoch}/{N_EPOCH}] | Source Loss: {total_loss:.4f} | Source Acc: {src_acc:.4f}%'
    tqdm.write(res)
    log_train.write(f"{epoch},{total_loss:.4f},{src_acc:.4f}\n")
    RESULT_SOURCE_TRAIN.append([epoch, total_loss, src_acc])
    return model


@torch.no_grad()
def eval_target(model, data_tar_train, data_tar_val, epoch):
    model.eval()
    correct_tar_train = 0
    for d, t in data_tar_train:
        d = d.to(DEVICE)
        t = t.to(DEVICE)
        correct_tar_train += model(d)[0].argmax(1).eq(t).sum().item()

    correct_tar_val = 0
    for d, t in data_tar_val:
        d = d.to(DEVICE)
        t = t.to(DEVICE)
        correct_tar_val += model(d)[0].argmax(1).eq(t).sum().item()

    tar_train_acc = 100.0 * correct_tar_train / len(data_tar_train.dataset)
    tar_val_acc = 100.0 * correct_tar_val / len(data_tar_val.dataset)

    res = f'Target Train Acc: {tar_train_acc:.4f}% | Target Val Acc: {tar_val_acc:.4f}%'
    tqdm.write(res)
    log_val.write(f"{epoch},{tar_train_acc:.4f},{tar_val_acc:.4f}\n")
    RESULT_TARGET_TRAIN.append([epoch, tar_train_acc])
    RESULT_TARGET_VAL.append([epoch, tar_val_acc])

    return tar_train_acc, tar_val_acc


if __name__ == '__main__':
    import data_loader

    rootdir = 'F:/data/'
    torch.manual_seed(1)

    # 加载数据
    data_src = data_loader.load_data(rootdir, 's', BATCH_SIZE[0])
    tar_full = data_loader.load_test(rootdir, 't', BATCH_SIZE[1]).dataset

    # 拆分数据集
    train_size = int(0.8 * len(tar_full))
    val_size = len(tar_full) - train_size
    tar_train, tar_val = random_split(tar_full, [train_size, val_size])

    data_tar_train = torch.utils.data.DataLoader(tar_train, BATCH_SIZE[1], shuffle=True, num_workers=0, drop_last=True)
    data_tar_val = torch.utils.data.DataLoader(tar_val, BATCH_SIZE[1], shuffle=False, num_workers=0)

    # 初始化模型
    model = DaNN.DaNN(n_class=10).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_WEIGHT)

    # 训练循环
    for epoch in tqdm(range(1, N_EPOCH + 1)):
        model = train(model, optimizer, epoch, data_src, data_tar_train)
        _, tar_val_acc = eval_target(model, data_tar_train, data_tar_val, epoch)
        tqdm.write('-' * 80)

    # 保存最终模型和日志
    torch.save(model.state_dict(), 'DANN_ViT.pth')
    log_train.close()
    log_val.close()

    np.savetxt('DANN_ViT_source_train.csv', np.array(RESULT_SOURCE_TRAIN), fmt='%.4f', delimiter=',')
    np.savetxt('DANN_ViT_target_train.csv', np.array(RESULT_TARGET_TRAIN), fmt='%.4f', delimiter=',')
    np.savetxt('DANN_ViT_target_val.csv', np.array(RESULT_TARGET_VAL), fmt='%.4f', delimiter=',')
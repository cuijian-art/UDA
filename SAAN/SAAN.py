import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import math
import ViT
import os
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.switch_backend('Agg')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
N_EPOCH = 100
NUM_CLASSES = 9
WEIGHT = 0.5

TSNE_SAVE_DIR = r'F:\pytorch_classification\DaNN\tsne_lmmd9'
os.makedirs(TSNE_SAVE_DIR, exist_ok=True)

RESULT_SOURCE_TRAIN = []
RESULT_TARGET = []


def set_paper_style():
    target_font = 'Times New Roman'
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if target_font not in available_fonts:
        print(f"⚠️ 警告：未找到 {target_font}，将使用衬线 fallback 字体 DejaVu Serif")
        used_font = 'DejaVu Serif'
    else:
        used_font = target_font

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [used_font],
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.grid': True,
        'figure.dpi': 100,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })
    print(f"✅ 绘图风格已设置，使用字体: {used_font}")


set_paper_style()


def rbf_mmd(x, y):
    x = x.mean(dim=0, keepdim=True)
    y = y.mean(dim=0, keepdim=True)
    return torch.mean((x - y) ** 2)


def local_mmd(source_features, source_labels, target_features, target_pseudo_labels):
    total_mmd = 0.0
    valid_class = 0
    for c in range(NUM_CLASSES):
        src_idx = torch.where(source_labels == c)[0]
        tar_idx = torch.where(target_pseudo_labels == c)[0]
        if len(src_idx) < 1 or len(tar_idx) < 1:
            continue
        total_mmd += rbf_mmd(source_features[src_idx], target_features[tar_idx])
        valid_class += 1
    return total_mmd / max(valid_class, 1)


def train(model, optimizer, epoch, data_src, data_tar_train):
    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_cls = 0.0
    total_domain = 0.0
    total_lmmd = 0.0
    correct_src = 0

    tar_iter = iter(data_tar_train)
    alpha = 1.0

    lambd = 2 / (1 + math.exp(-10 * (epoch) / N_EPOCH)) - 1

    for data, target in data_src:
        try:
            x_tar, _ = next(tar_iter)
        except StopIteration:
            tar_iter = iter(data_tar_train)
            x_tar, _ = next(tar_iter)

        bs = min(data.size(0), x_tar.size(0))
        data, target = data[:bs], target[:bs]
        x_tar = x_tar[:bs]

        data, target, x_tar = data.to(DEVICE), target.to(DEVICE), x_tar.to(DEVICE)

        src_label, src_domain = model(data, alpha)
        tar_label, tar_domain = model(x_tar, alpha)

        loss_cls = criterion_cls(src_label, target)
        loss_domain = criterion_domain(src_domain, torch.zeros(bs).long().to(DEVICE)) + \
                      criterion_domain(tar_domain, torch.ones(bs).long().to(DEVICE))

        feat_src = model.extract_feature(data)
        feat_tar = model.extract_feature(x_tar)
        loss_lmmd = local_mmd(feat_src, target, feat_tar, tar_label.argmax(1))
        weighted_lmmd = WEIGHT * lambd * loss_lmmd

        loss = loss_cls + loss_domain + weighted_lmmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_domain += loss_domain.item()
        total_lmmd += weighted_lmmd.item()
        correct_src += src_label.argmax(1).eq(target).sum().item()

    avg_loss = total_loss / len(data_src)
    avg_cls = total_cls / len(data_src)
    avg_domain = total_domain / len(data_src)
    avg_lmmd = total_lmmd / len(data_src)
    src_acc = 100.0 * correct_src / len(data_src.dataset)

    print(
        f'[训练] Epoch: {epoch:03d} | λ: {lambd:.4f} | Loss: {avg_loss:.4f} | Cls: {avg_cls:.4f} | Domain: {avg_domain:.4f} | LMMD: {avg_lmmd:.4f} | Acc: {src_acc:.2f}%')
    RESULT_SOURCE_TRAIN.append([epoch, avg_loss, avg_cls, avg_domain, avg_lmmd, src_acc])
    return model


@torch.no_grad()
def eval_target(model, data_tar_val):
    model.eval()
    criterion_cls = nn.CrossEntropyLoss()
    total_tar_loss = 0.0
    correct_tar = 0

    for data, target in data_tar_val:
        data, target = data.to(DEVICE), target.to(DEVICE)
        pred, _ = model(data)

        loss = criterion_cls(pred, target)
        total_tar_loss += loss.item()
        correct_tar += pred.argmax(1).eq(target).sum().item()

    avg_tar_loss = total_tar_loss / len(data_tar_val)
    tar_acc = 100.0 * correct_tar / len(data_tar_val.dataset)

    print(f'[测试] Target Loss: {avg_tar_loss:.4f} | Target Acc: {tar_acc:.2f}%')
    RESULT_TARGET.append([avg_tar_loss, tar_acc])
    return avg_tar_loss, tar_acc


def plot_tsne(model, src_loader, tar_loader, device, save_dir, epoch, tar_val_acc, num_samples=300):
    model.eval()
    features = []
    labels = []
    domains = []

    src_count = 0
    with torch.no_grad():
        for data, label in tqdm(src_loader, desc=f'[t-SNE Epoch {epoch}] Extracting Source'):
            if num_samples is not None and src_count >= num_samples:
                break
            data = data.to(device)
            feat = model.extract_feature(data)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
            labels.append(label.cpu().numpy())
            domains.append(np.zeros(len(label)))
            src_count += len(label)
        if num_samples is not None:
            trunc = num_samples - (src_count - len(features[-1]))
            features[-1] = features[-1][:trunc]
            labels[-1] = labels[-1][:trunc]
            domains[-1] = domains[-1][:trunc]
    tar_count = 0
    with torch.no_grad():
        for data, label in tqdm(tar_loader, desc=f'[t-SNE Epoch {epoch}] Extracting Target'):
            if num_samples is not None and tar_count >= num_samples:
                break
            data = data.to(device)
            feat = model.extract_feature(data)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
            labels.append(label.cpu().numpy())
            domains.append(np.ones(len(label)))
            tar_count += len(label)
        if num_samples is not None:
            trunc = num_samples - (tar_count - len(features[-1]))
            features[-1] = features[-1][:trunc]
            labels[-1] = labels[-1][:trunc]
            domains[-1] = domains[-1][:trunc]

    # 降维
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    domains = np.concatenate(domains, axis=0)
    print(f'[t-SNE Epoch {epoch}] Running dimensionality reduction...')
    tsne = TSNE(n_components=2, perplexity=min(30, num_samples // 5), n_iter=1500,
                early_exaggeration=12, random_state=42, n_jobs=-1)
    features_2d = tsne.fit_transform(features)

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 5.2))
    src_idx = domains == 0
    tar_idx = domains == 1

    cmap = plt.get_cmap('tab20c', 9)

    ax.scatter(
        features_2d[src_idx, 0], features_2d[src_idx, 1],
        c=labels[src_idx], cmap=cmap,
        marker='o', s=40, alpha=0.65,
        edgecolor='#2D2D2D', linewidth=0.4,
        label='Source',
        zorder=5
    )
    ax.scatter(
        features_2d[tar_idx, 0], features_2d[tar_idx, 1],
        c=labels[tar_idx], cmap=cmap,
        marker='^', s=40, alpha=0.85,
        edgecolor='#2D2D2D', linewidth=0.4,
        label='Target',
        zorder=5
    )

    ax.grid(True, color='#E5E5E5', linestyle='--', linewidth=0.7, zorder=0)
    ax.set_xlabel('t-SNE Dimension 1', labelpad=8)
    ax.set_ylabel('t-SNE Dimension 2', labelpad=8)

    # 保存高清图片
    save_filename = f"lmmd_tsne_epoch_{epoch:03d}_tar_val_{tar_val_acc:.4f}.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f'[t-SNE Epoch {epoch}] ✅ Saved high-resolution PNG:\n  {save_path}')


if __name__ == '__main__':
    import data_loader

    rootdir = 'F:\data/'

    data_src = data_loader.load_data(rootdir, 's', BATCH_SIZE)
    tar_dataset = data_loader.load_test(rootdir, 't', BATCH_SIZE).dataset

    tar_train, tar_val = random_split(tar_dataset,
                                      [int(0.8 * len(tar_dataset)), len(tar_dataset) - int(0.8 * len(tar_dataset))])
    data_tar_train = torch.utils.data.DataLoader(tar_train, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    data_tar_val = torch.utils.data.DataLoader(tar_val, BATCH_SIZE, shuffle=False, num_workers=2)

    data_tar_full_tsne = torch.utils.data.DataLoader(tar_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = DaNN.DaNN(NUM_CLASSES).to(DEVICE)

    optimizer = optim.AdamW(
        [
            {'params': model.vit.parameters(), 'lr': 1e-5},
            {'params': model.label_predictor.parameters(), 'lr': 1e-4},
            {'params': model.domain_discriminator.parameters(), 'lr': 1e-4}
        ],
        weight_decay=1e-4
    )

    for epoch in range(1, N_EPOCH + 1):
        model = train(model, optimizer, epoch, data_src, data_tar_train)
        _, tar_val_acc = eval_target(model, data_tar_val)
        print('-' * 130)

        if epoch % 2 == 0:
            plot_tsne(
                model,
                data_src,
                data_tar_full_tsne,
                DEVICE,
                save_dir=TSNE_SAVE_DIR,
                epoch=epoch,
                tar_val_acc=tar_val_acc,
                num_samples=300
            )
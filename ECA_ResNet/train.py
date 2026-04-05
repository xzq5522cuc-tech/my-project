import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import xlwt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from eca_resnet import eca_resnet34

def imshow(tensor_img, title=None, save_path='debug_preview.png'):
    """显示并保存单个Tensor图像（反标准化后）"""
    img = tensor_img.numpy().transpose((1, 2, 0))
    mean = np.array([0.456, 0.432, 0.415])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"预览图像已保存至 {save_path}")


def plot_training_history(train_losses, test_losses, train_accs, test_accs, save_path='training_history.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, 'b-', label='Train Loss')
    axes[0].plot(test_losses, 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(train_accs, 'b-', label='Train Accuracy')
    axes[1].plot(test_accs, 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"训练曲线图已保存至 {save_path}")


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"混淆矩阵图已保存至 {save_path}")


def calculate_metrics(all_labels, all_preds, num_classes):
    """
    根据真实标签和预测标签计算各项评估指标
    返回字典包含：混淆矩阵、总体准确率、各类别精确率/召回率/F1、宏平均等
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_lbl, pred_lbl in zip(all_labels, all_preds):
        conf_matrix[true_lbl][pred_lbl] += 1

    precision = []
    recall = []
    f1 = []
    support = []

    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - tp
        fp = conf_matrix[:, i].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(tp + fn)

    overall_acc = np.trace(conf_matrix) / conf_matrix.sum()
    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)

    return {
        'confusion_matrix': conf_matrix,
        'overall_accuracy': overall_acc,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'total_samples': len(all_labels)
    }


class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, verbose=True, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} -> {self.best_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)


class CustomDataParallel(nn.DataParallel):
    """解决DataParallel无法直接访问module属性的问题"""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


# ----------------------------- 训练与评估核心函数 -----------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(data_loader, desc=f"Train Epoch {epoch}", file=sys.stdout)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        pbar.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(data_loader, desc=f"Val Epoch {epoch}", file=sys.stdout)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    metrics = calculate_metrics(all_labels, all_preds, num_classes)
    return avg_loss, avg_acc, metrics


# ----------------------------- 主函数 -----------------------------
def main():
    # 环境与设备
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径与预处理
    data_root = "/data/users/xzq/nilm/dataset_new"
    assert os.path.exists(data_root), f"数据路径不存在: {data_root}"

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # 图像标准化参数（与预训练模型匹配）
    mean = [0.456, 0.432, 0.415]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # 加载数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    print(f"训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")

    # 类别映射
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"类别数: {num_classes}")
    for name, idx in class_to_idx.items():
        print(f"  {idx}: {name}")

    # 保存类别映射
    with open('class_indices_plaid.json', 'w') as f:
        json.dump(idx_to_class, f, indent=4)

    # 预览一张训练图像
    sample_img, sample_label = train_dataset[0]
    imshow(sample_img, title=f"Sample: {idx_to_class[sample_label]}", save_path='debug_preview.png')
    print(f"图像Tensor形状: {sample_img.shape}, 像素范围: [{sample_img.min():.3f}, {sample_img.max():.3f}]")

    # DataLoader
    batch_size = 64
    num_workers = min(os.cpu_count(), batch_size, 8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # 模型
    model = eca_resnet34()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 块 GPU")
        model = CustomDataParallel(model)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

    # 优化器、调度器
    lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    # 可选：加载预训练权重
    pretrained_path = "/data/users/xzq/nilm/ECA_ResNet/weight/ECA_ResNet34_Myself.pth"
    if os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("未找到预训练权重，从头开始训练")

    # 训练记录
    epochs = 60
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    save_path = "/data/users/xzq/nilm/ECA_ResNet/weight/ECA_ResNet34_best.pth"

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    f1_scores = []

    # Excel记录
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet_train = workbook.add_sheet('Train_data')
    headers = ['epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'lr', 'Best_Val_Acc', 'Best_F1']
    for col, h in enumerate(headers):
        sheet_train.write(0, col, h)

    # TensorBoard
    writer = SummaryWriter()

    # 早停
    early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=True, path=save_path)

    # 训练循环
    for epoch in range(1, epochs + 1):
        print(f"\n========== Epoch {epoch}/{epochs} ==========")
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, device, epoch, num_classes)

        # 保存记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        f1_scores.append(val_metrics['macro_f1'])

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 更新最佳指标
        if val_acc > best_acc:
            best_acc = val_acc
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            # 保存完整检查点（包含优化器状态等）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, CustomDataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'f1_macro': val_metrics['macro_f1'],
                'class_to_idx': class_to_idx,
                'num_classes': num_classes,
                'batch_size': batch_size,
                'learning_rate': lr
            }, save_path)

        # 早停检测
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stop training.")
            break

        # 写入Excel
        sheet_train.write(epoch, 0, epoch)
        sheet_train.write(epoch, 1, float(train_loss))
        sheet_train.write(epoch, 2, float(train_acc))
        sheet_train.write(epoch, 3, float(val_loss))
        sheet_train.write(epoch, 4, float(val_acc))
        sheet_train.write(epoch, 5, float(current_lr))
        sheet_train.write(epoch, 6, float(best_acc))
        sheet_train.write(epoch, 7, float(best_f1))

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1_macro/val', val_metrics['macro_f1'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Macro F1:   {val_metrics['macro_f1']:.4f}")

    # 绘制训练曲线
    plot_training_history(train_losses, val_losses, train_accs, val_accs, 'training_history.png')

    writer.close()


if __name__ == '__main__':
    main()
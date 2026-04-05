import os
import json
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eca_resnet import eca_resnet34
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 核心工具函数 --------------------
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """打印进度条"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

def calculate_metrics(conf_matrix):
    """根据混淆矩阵计算各项评估指标"""
    n_classes = conf_matrix.shape[0]
    precision_per_class, recall_per_class, f1_per_class = [], [], []

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    return {
        'confusion_matrix': conf_matrix,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': np.sum(conf_matrix, axis=1),
        'overall_accuracy': np.trace(conf_matrix) / np.sum(conf_matrix),
        'macro_precision': np.mean(precision_per_class),
        'macro_recall': np.mean(recall_per_class),
        'macro_f1': np.mean(f1_per_class)
    }

def print_metrics(metrics, class_names):
    """精简版指标打印"""
    print("\n" + "=" * 70)
    print("整体评估指标")
    print("=" * 70)
    print(f"总体准确率: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy'] * 100:.2f}%)")
    print(f"宏平均精确率: {metrics['macro_precision']:.4f}")
    print(f"宏平均召回率: {metrics['macro_recall']:.4f}")
    print(f"宏平均F1分数: {metrics['macro_f1']:.4f}")
    print(f"\n总样本数: {np.sum(metrics['confusion_matrix'])}")

    print("\n" + "-" * 70)
    print("各类别详细指标")
    print("-" * 70)
    print(f"{'类别':<25} {'样本数':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {metrics['support_per_class'][i]:<8} "
              f"{metrics['precision_per_class'][i]:<8.4f} "
              f"{metrics['recall_per_class'][i]:<8.4f} "
              f"{metrics['f1_per_class'][i]:<8.4f}")

def analyze_errors(y_true, y_pred, img_paths, class_names):
    """仅生成数据，不打印"""
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]

    error_analysis = {}
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true == i)[0]
        class_errors = sum(y_true[class_indices] != y_pred[class_indices])
        class_total = len(class_indices)

        if class_total > 0:
            error_rate = class_errors / class_total * 100
            error_analysis[class_name] = {
                'total_samples': class_total, 'errors': class_errors, 'error_rate': error_rate,
                'misclassifications': {}
            }
            for idx in class_indices:
                if y_true[idx] != y_pred[idx]:
                    pred_class = class_names[y_pred[idx]]
                    error_analysis[class_name]['misclassifications'][pred_class] = \
                        error_analysis[class_name]['misclassifications'].get(pred_class, 0) + 1

    error_details = []
    for idx in error_indices:
        error_details.append({
            'image_path': img_paths[idx],
            'filename': os.path.basename(img_paths[idx]),
            'true_class': class_names[y_true[idx]],
            'pred_class': class_names[y_pred[idx]],
            'true_idx': int(y_true[idx]),
            'pred_idx': int(y_pred[idx])
        })

    return error_analysis, error_details

def save_confusion_matrix_plot(conf_matrix, class_names, save_path):
    """保存混淆矩阵热力图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    percentages = conf_matrix.astype(float) / row_sums * 100
    recalls = np.diag(conf_matrix) / conf_matrix.sum(axis=1) * 100
    precisions = np.diag(conf_matrix) / conf_matrix.sum(axis=0) * 100

    annot_text = np.empty_like(conf_matrix, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            annot_text[i, j] = f"{count}\n({percentages[i, j]:.1f}%)" if count > 0 else ''

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(percentages, annot=annot_text, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    y_labels = [f"{name}\n召回率: {recall:.1f}%" for name, recall in zip(class_names, recalls)]
    x_labels = [f"{name}\n精确率: {precision:.1f}%" for name, precision in zip(class_names, precisions)]
    ax.set_yticklabels(y_labels, rotation=0), ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_title('混淆矩阵 (计数 + 百分比)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('预测类别', fontsize=14)
    ax.set_ylabel('真实类别', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_reports(metrics, error_analysis, error_details, class_names, conf_matrix, save_dir):
    """保存全套 CSV 报告"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. 混淆矩阵 CSV
    pd.DataFrame(conf_matrix, 
                 index=[f"True_{n}" for n in class_names],
                 columns=[f"Pred_{n}" for n in class_names])\
      .to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))

    # 2. 类别指标 CSV
    pd.DataFrame({
        '类别': class_names,
        '样本数': metrics['support_per_class'],
        '精确率': metrics['precision_per_class'],
        '召回率': metrics['recall_per_class'],
        'F1分数': metrics['f1_per_class']
    }).to_csv(os.path.join(save_dir, 'class_metrics.csv'), index=False, encoding='utf-8-sig')

    # 3. 错误分析 CSV
    if error_analysis:
        error_df_data = []
        for class_name, analysis in error_analysis.items():
            row = {
                '类别': class_name,
                '总样本数': analysis['total_samples'],
                '错误数': analysis['errors'],
                '错误率(%)': analysis['error_rate']
            }
            if analysis['misclassifications']:
                sorted_mis = sorted(analysis['misclassifications'].items(), key=lambda x: x[1], reverse=True)
                for i, (wrong_class, count) in enumerate(sorted_mis[:3], 1):
                    row[f'误判{i}_类别'] = wrong_class
                    row[f'误判{i}_次数'] = count
                    row[f'误判{i}_占比(%)'] = count / analysis['errors'] * 100
            error_df_data.append(row)
        pd.DataFrame(error_df_data).to_csv(os.path.join(save_dir, 'error_analysis.csv'), 
                                             index=False, encoding='utf-8-sig')

    # 4. 错误样本详情 CSV
    if error_details:
        pd.DataFrame(error_details).to_csv(os.path.join(save_dir, 'error_samples_details.csv'), 
                                            index=False, encoding='utf-8-sig')

def get_image_paths_and_labels(root_dir, class_to_idx):
    """从文件夹结构获取图像路径和标签"""
    img_paths, labels = [], []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir) and class_name in class_to_idx:
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_paths.append(os.path.join(class_dir, f))
                    labels.append(class_to_idx[class_name])
    return img_paths, labels

def main():
    test_imgs_root = r"/data/users/xzq/nilm/dataset_new/test"
    class_indices_json = r"/data/users/xzq/nilm/dataset_new/class_indices.json"
    model_weights_path = r"/data/users/xzq/nilm/ECA_ResNet/weight/best/ECA_ResNet34_best.pth"
    report_save_dir = r"/data/users/xzq/nilm/ECA_ResNet/evaluation_reports"
    
    input_size = (244, 244)
    mean = [0.456, 0.432, 0.415]
    std = [0.229, 0.224, 0.225]
    batch_size = 32
    # ===========================================================

    # 1. 设备初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 数据预处理
    data_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 3. 加载类别映射
    assert os.path.exists(class_indices_json), f"类别文件不存在: {class_indices_json}"
    with open(class_indices_json, "r") as f:
        class_to_idx = json.load(f)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
    
    print(f"\n加载了 {len(class_names)} 个类别:")
    for i, name in enumerate(class_names):
        print(f"  类别 {i}: {name} ({class_to_idx[name]})")

    # 4. 加载测试图像
    assert os.path.exists(test_imgs_root), f"测试集路径不存在: {test_imgs_root}"
    print(f"\n正在从 {test_imgs_root} 加载图像...")
    img_path_list, true_labels = get_image_paths_and_labels(test_imgs_root, class_to_idx)
    print(f"找到 {len(img_path_list)} 张图像")

    # 5. 加载模型
    model = eca_resnet34()
    assert os.path.exists(model_weights_path), f"权重文件不存在: {model_weights_path}"
    print(f"\n加载模型权重: {model_weights_path}")
    checkpoint = torch.load(model_weights_path, map_location=device)

    # 确定类别数并修改全连接层
    if isinstance(checkpoint, dict) and 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        num_classes = checkpoint.get('num_classes', len(class_to_idx))
        print(f"从检查点获取类别数: {num_classes}")
    else:
        num_classes = len(class_names)
        print(f"从JSON文件获取类别数: {num_classes}")
    
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, num_classes)
    print(f"设置模型为 {num_classes} 类")

    # 加载权重（兼容 DataParallel）
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if isinstance(state_dict, dict) and list(state_dict.keys())[0].startswith('module.'):
        print("检测到 DataParallel 前缀，正在移除...")
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    print("使用 'model_state_dict' 中的权重")
    
    model.load_state_dict(state_dict, strict=False)
    print("模型权重加载完成")
    model.to(device)
    model.eval()

    # 6. 批量预测
    print(f"\n开始预测，共 {len(img_path_list)} 张图像，批次大小: {batch_size}")
    all_preds = []
    total_batches = (len(img_path_list) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(img_path_list))
            
            img_list = []
            for idx in range(start_idx, end_idx):
                try:
                    img = Image.open(img_path_list[idx])
                    img_list.append(data_transform(img))
                except Exception as e:
                    img_list.append(torch.zeros(3, *input_size))
            
            if img_list:
                batch_img = torch.stack(img_list, dim=0).to(device)
                output = model(batch_img).cpu()
                _, preds = torch.max(output, dim=1)
                all_preds.extend(preds.numpy())
            
            print_progress_bar(batch_idx + 1, total_batches,
                               prefix='预测进度:',
                               suffix=f'批次 {batch_idx + 1}/{total_batches}')

    all_preds = np.array(all_preds)
    all_labels = np.array(true_labels)
    print(f"\n预测完成! 预测样本数: {len(all_preds)}, 真实标签数: {len(all_labels)}")

    # 7. 计算指标并打印
    n_classes = len(class_names)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        conf_matrix[t, p] += 1
    
    metrics = calculate_metrics(conf_matrix)
    print_metrics(metrics, class_names)

    # 8. 生成报告数据（不打印）
    error_analysis, error_details = analyze_errors(all_labels, all_preds, img_path_list, class_names)

    # 9. 保存报告
    save_reports(metrics, error_analysis, error_details, class_names, conf_matrix, report_save_dir)
    save_confusion_matrix_plot(conf_matrix, class_names, 
                               os.path.join(report_save_dir, 'confusion_matrix.png'))
    
    # 10. 最终提示（完全匹配你的要求）
    print(f"\n详细报告已保存到目录: {report_save_dir}")
    print(f"混淆矩阵图已保存到: {os.path.join(report_save_dir, 'confusion_matrix.png')}；"
          f"{os.path.join(report_save_dir, 'error_samples_details.csv')}；"
          f"{os.path.join(report_save_dir, 'error_analysis.csv')}")

if __name__ == '__main__':
    main()
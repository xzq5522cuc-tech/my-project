import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split

def split_plaid_stratified(image_root_dir, meta_json_path, output_dir,
                           train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                           random_seed=42):
    """
    按类别分层划分PLAID数据集（基于设备实例）
    每个类别内的设备ID按比例随机分配到 train/val/test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(random_seed)

    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # 加载元数据：id -> 设备类型
    with open(meta_json_path, 'r') as f:
        meta_data = json.load(f)
    id_to_type = {str(item["id"]): item["meta"]["appliance"]["type"] for item in meta_data}

    # 获取所有存在的设备文件夹（ID必须在元数据中）
    all_ids = [d for d in os.listdir(image_root_dir)
               if os.path.isdir(os.path.join(image_root_dir, d)) and d in id_to_type]

    # 按类型分组
    type_to_ids = {}
    for id_ in all_ids:
        t = id_to_type[id_]
        type_to_ids.setdefault(t, []).append(id_)

    print("设备类型分布（实例数）:")
    for t, ids in type_to_ids.items():
        print(f"  {t}: {len(ids)} 个设备")

    train_ids = []
    val_ids = []
    test_ids = []

    # 对每个类型分别划分
    for t, ids in type_to_ids.items():
        n = len(ids)
        if n < 3:
            # 样本太少，全部放入训练集（避免验证/测试集无代表性）
            train_ids.extend(ids)
            print(f"  警告: 类别 '{t}' 只有 {n} 个实例，全部放入训练集")
            continue

        # 按比例划分：先分出测试集
        ids_trainval, ids_test = train_test_split(
            ids, test_size=test_ratio, random_state=random_seed, shuffle=True
        )
        # 从 trainval 中按比例分出验证集（相对于原始训练+验证的比例）
        val_ratio_in_trainval = val_ratio / (train_ratio + val_ratio)
        ids_train, ids_val = train_test_split(
            ids_trainval, test_size=val_ratio_in_trainval, random_state=random_seed, shuffle=True
        )
        train_ids.extend(ids_train)
        val_ids.extend(ids_val)
        test_ids.extend(ids_test)

    print(f"\n划分结果（实例数）:")
    print(f"  训练集: {len(train_ids)}")
    print(f"  验证集: {len(val_ids)}")
    print(f"  测试集: {len(test_ids)}")

    # 复制图像到对应目录
    for split_name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = os.path.join(output_dir, split_name)
        for id_ in id_list:
            category = id_to_type[id_]
            cat_dir = os.path.join(split_dir, category)
            os.makedirs(cat_dir, exist_ok=True)
            src_dir = os.path.join(image_root_dir, id_)
            for img_file in os.listdir(src_dir):
                if img_file.lower().endswith('.png'):
                    src_path = os.path.join(src_dir, img_file)
                    # 使用 设备ID_原文件名 避免重名
                    dst_name = f"{id_}_{img_file}"
                    dst_path = os.path.join(cat_dir, dst_name)
                    shutil.copy2(src_path, dst_path)
        print(f"  {split_name}: 已复制 {len(id_list)} 个设备的图像")

    # 保存类别映射（按字母排序）
    class_names = sorted(type_to_ids.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    with open(os.path.join(output_dir, "class_indices.json"), 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    print("\n数据集划分完成！")


if __name__ == "__main__":
    image_root_dir = "/data/users/xzq/nilm/data/Colorful_VI_Trajectory"
    meta_json_path = "/data/users/xzq/nilm/plaid 2014/meta_2014.json"
    output_dir = "/data/users/xzq/nilm/dataset_new"

    split_plaid_stratified(
        image_root_dir, meta_json_path, output_dir,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_seed=42
    )


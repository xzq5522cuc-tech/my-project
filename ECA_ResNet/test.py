import os

train_dir = "/data/users/xzq/nilm/dataset_new_2/train"
test_dir = "/data/users/xzq/nilm/dataset_new_2/test"

train_ids = set()
for root, dirs, files in os.walk(train_dir):
    for f in files:
        if f.endswith('.png'):
            # 假设文件名格式为 "设备ID_cycle_X.png"
            id_ = f.split('_')[0]
            train_ids.add(id_)

test_ids = set()
for root, dirs, files in os.walk(test_dir):
    for f in files:
        if f.endswith('.png'):
            id_ = f.split('_')[0]
            test_ids.add(id_)

overlap = train_ids & test_ids
print(f"重叠的设备ID数量: {len(overlap)}")
if overlap:
    print("重叠ID示例:", list(overlap)[:10])
# 电器负荷识别系统
本项目基于 **PLAID 2014** 公开数据集，对每个电器设备的电压/电流原始波形进行预处理，提取稳态运行周期，并生成彩色 3D V-I 轨迹图像。预处理后的图像可用于后续深度学习模型（如 ECA-ResNet）进行电器类型识别。

PLAID 数据集以 30 kHz 采样率记录了 11 类家用电器（空调、风扇、冰箱、微波炉等）的电压/电流波形。但由于原始数据长度不一、启动暂态干扰等因素，直接使用完整波形会导致模型性能下降。本预处理模块的核心目标是：

- **自动检测稳态运行区间**（有功功率稳定、波形周期一致）
- **提取连续的若干个完整工频周期**（默认 15 个周期）
- **对齐电压正过零点**，消除相位随机性
- **计算电气特征**（功率因数、三次谐波幅值等）
- **生成彩色 3D V-I 轨迹图**（颜色由瞬时运动方向、功率因数、谐波含量共同决定）

预处理后的图像将作为深度神经网络的输入，用于电器识别任务。

## 2. 数据预处理流程

plaid 2014/                  # 数据集
├─ 2014/                     # 【必须配置】PLAID 2014原始数据集文件夹
│  └─ 1.csv、2.csv...1876.csv  # 官方分表数据，单电器原始电压/电流CSV文件
├─ meta_2014.json     

process_data/              # 生成3D轨迹
├─ __pycache__/    
├─ steady_plots/          # 提取的稳态周期情况
├─ utilities.py           # 通用工具函数：相位对齐、峰值检测等
├─ steady_samples.py      # 稳态样本提取：RMS计算、稳态区间判定、有效周期提取核心逻辑
├─ harmonics.py           # 谐波分析：FFT频域计算、各次谐波幅值提取、谐波特征构建
├─ process_data.py        # 数据预处理核心：元数据解析、原始CSV读取、电器信息字典构建
├─ split.py               # 划分数据集
└─ VI_Trajectory.py       # 【主运行脚本】V-I轨迹生成全流程

//V-I_Trajectory.py：原始开源VI_Trajectory处理脚本

./data/Colorful_VI_Trajectory 处理好的数据集
./data/users/xzq/nilm/dataset_new 最终划分好的数据集

#训练
ECA_ResNet/                      # 模型训练根目录                            
├─ train.py                      # 模型训练全流程实现
├─ training_history.png          # 训练过程 loss/accuracy 曲线可视化结果


#测试
ECA_ResNet/                      # 模型训练根目录
├─ batch_predict.py              # test数据集上批量评估
├─ reports                       # 评估报告


在测试集上达到 **97.5% 的总体准确率**，宏平均 F1 为 0.954。评估脚本自动生成混淆矩阵（含召回率/精确率标注）、各类别详细指标（CSV）及错误样本分析报告。

**核心文件**：`process_data/VI_Trajectory.py`（生成轨迹图）、`process_data/split.py`（分层划分）、`ECA_ResNet/train.py`（训练）、`ECA_ResNet/batch_predict.py`（评估）。环境依赖见 `environment.yml`，plaid_2014.zip数据集请从 Releases 下载




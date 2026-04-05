import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


def normalize_data(column_data):
    """
    数据归一化函数：将数据缩放到 [0, 1] 区间
    参数:
        column_data: 待归一化的一维数据列
    返回:
        归一化后的数据
    """
    return (column_data - column_data.min()) / (column_data.max() - column_data.min())


def compute_power(voltage, current):
    """
    计算瞬时功率
    参数:
        voltage: 电压数组
        current: 电流数组
    返回:
        瞬时功率数组
    """
    return voltage * current


def compute_power_factor(voltage, current):
    """
    计算功率因数 (PF)
    参数:
        voltage: 电压数组
        current: 电流数组
    返回:
        功率因数值
    公式:
        PF = 有功功率(P) / 视在功率(S)
        其中 P = mean(voltage * current), S = V_rms * I_rms
    """
    P_active = np.mean(voltage * current)  # 有功功率
    V_rms = np.sqrt(np.mean(voltage**2))   # 电压有效值
    I_rms = np.sqrt(np.mean(current**2))   # 电流有效值
    S_apparent = V_rms * I_rms             # 视在功率
    PF = P_active / S_apparent              # 功率因数
    return PF


def compute_hue(voltage, current):
    """
    计算色调 (Hue) 分量，用于 HSV 颜色空间
    参数:
        voltage: 电压数组
        current: 电流数组
    返回:
        归一化到 [0, 1] 的色调数组
    原理:
        通过电压和电流的微分变化率计算角度，映射为色调
    """
    H = np.zeros_like(voltage)
    for j in range(len(voltage) - 1):
        # 计算相邻点的电压和电流变化量
        delta_v = voltage[j + 1] - voltage[j]
        delta_i = current[j + 1] - current[j]

        # 计算变化率的角度 (弧度制)
        angle = np.arctan2(delta_v, delta_i)

        # 将角度从 [-π, π] 归一化到 [0, 1]
        normalized_angle = (angle + np.pi) / (2 * np.pi)
        H[j] = normalized_angle
    H[-1] = H[-2]  # 最后一个点复用前一个点的值
    return H


def compute_third_harmonic(data, sample_rate):
    """
    计算信号的三次谐波幅值 (针对 60Hz 基波)
    参数:
        data: 输入信号数组
        sample_rate: 采样率
    返回:
        三次谐波的幅值
    原理:
        使用 FFT (快速傅里叶变换) 进行频域分析
    """
    fft_result = np.fft.fft(data)  # 执行 FFT

    # 计算每个 FFT 点对应的频率
    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)

    # 三次谐波频率 (假设基波为 60Hz)
    third_harmonic_freq = 3 * 60

    # 找到最接近三次谐波频率的索引
    third_harmonic_index = np.argmin(np.abs(freqs - third_harmonic_freq))

    # 获取该频率点的幅值
    third_harmonic_magnitude = np.abs(fft_result[third_harmonic_index])
    return third_harmonic_magnitude


def plot_3d_power_hue_vi_trajectory(data, save_path, sample_rate):
    """
    绘制 3D 彩色 VI 轨迹图 (电压-电流-时间)
    参数:
        data: 包含电流和电压的 DataFrame
        save_path: 图片保存路径
        sample_rate: 采样率
    颜色映射:
        H (色调): 由 VI 微分轨迹决定
        S (饱和度): 由功率因数决定
        V (明度): 由电压三次谐波决定
    """
    voltage = data.iloc[:, 1].values  # 提取电压 (第2列)
    current = data.iloc[:, 0].values  # 提取电流 (第1列)
    # 生成时间戳 (假设总时长为 16.65ms，约一个 60Hz 周期)
    time_interval = 16.65 / len(voltage)
    time_stamps = np.arange(0, 16.65, time_interval)

    # 计算 HSV 颜色空间的三个分量
    PF = compute_power_factor(voltage, current)  # 功率因数
    H = compute_hue(voltage, current)             # 色调
    S = np.ones_like(H) * (0.5 + 0.5 * PF)       # 饱和度 (与 PF 正相关)
    # 明度 (基于三次谐波归一化)
    V_value = compute_third_harmonic(voltage, sample_rate) / np.max(compute_third_harmonic(voltage, sample_rate))
    V = np.full_like(H, V_value)

    # 将 HSV 颜色转换为 RGB 颜色 (Matplotlib 绘图需要)
    HSV = np.stack((H, S, V), axis=-1)
    RGB = colors.hsv_to_rgb(HSV)

    # 创建 3D 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 逐段绘制轨迹，每一段使用对应的 RGB 颜色
    for i in range(1, len(voltage)):
        ax.plot(voltage[i - 1:i + 1], current[i - 1:i + 1], time_stamps[i - 1:i + 1], color=RGB[i])

    # 设置坐标轴标签
    ax.set_xlabel("Normalized Voltage/(V)")
    ax.set_ylabel("Normalized Current/(A)")
    ax.set_zlabel("Time (ms)")
    plt.savefig(save_path)  # 保存图片
    plt.close()              # 关闭绘图释放内存

def sort_key(filename):
    """
    (未实际使用) 文件名排序辅助函数：提取文件名中的数字
    """
    numbers = [int(s) for s in filename.split('.') if s.isdigit()]
    return numbers[0] if numbers else -1


# --- 主程序开始 ---

# 输入 CSV 文件夹路径
csv_folder_path = r'/data/users/xzq/nilm/plaid 2014/2014'

# 输出图片保存的根文件夹路径
base_save_folder_path = r'/data/users/xzq/nilm/data/Colorful VI Trajectory'

# 采样率设置
sample_rate = 30000

# 获取文件夹下所有 CSV 文件
all_csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
# 按文件名中的数字大小排序文件
sorted_csv_files = sorted(all_csv_files, key=lambda f: int(''.join(filter(str.isdigit, f))))


# 循环处理每一个 CSV 文件
for idx, filename in enumerate(sorted_csv_files):

    # 提取文件名中的数字作为子文件夹名
    folder_number = int(''.join(filter(str.isdigit, filename)))
    print(f"Processing file {idx + 1}/{len(sorted_csv_files)}: {filename} into folder {folder_number}")

    try:
            # 创建保存图片的子文件夹
            save_folder_path = os.path.join(base_save_folder_path, str(folder_number))
            os.makedirs(save_folder_path, exist_ok=True)

            # 读取 CSV 文件 (无表头)
            csv_file_path = os.path.join(csv_folder_path, filename)
            data = pd.read_csv(csv_file_path, header=None)

            # 对电流 (第0列) 和 电压 (第1列) 分别进行归一化
            data.iloc[:, 0] = normalize_data(data.iloc[:, 0])
            data.iloc[:, 1] = normalize_data(data.iloc[:, 1])

            # 按周期分割数据 (假设每个周期 500 个采样点)
            samples_per_cycle = 500
            number_of_cycles = len(data) // samples_per_cycle

            # 循环处理每一个周期的数据并绘图
            for i in range(number_of_cycles):
                start = i * samples_per_cycle
                end = (i + 1) * samples_per_cycle
                cycle_data = data.iloc[start:end, :]
                save_path = os.path.join(save_folder_path, f"VI Trajectory {i + 1}.png")
                plot_3d_power_hue_vi_trajectory(cycle_data, save_path, sample_rate)

            print(f"Finished processing {filename} into folder {folder_number}")

    except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
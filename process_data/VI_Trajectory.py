import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
# 导入自定义工具模块（需确保这些文件在同一目录下）
from utilities import shift_phase, count_progress, find_peaks_rms, acronym_maker
from steady_samples import generate_rms
from process_data import metadata_submetered, steady_samples_submetered, construct_harmonics_dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import csv
# -------------------- 全局配置参数 --------------------
FS = 30000                     # 采样率：30 kHz（与 PLAID 数据集一致）
BASE_FREQ = 60                 # 电网基频：60 Hz（PLAID 为美国数据集）
CYCLES_FOR_AVG = 15            # 从稳态段提取的周期数（10~20 个周期较稳定）
POWER_STABLE_THRESHOLD = 0.02  # 功率稳定判定阈值（相对变化 2% 以内视为稳态）
CONSECUTIVE_WINDOWS = 5        # 判定稳态所需的连续稳定窗口数
WINDOW_LEN = int(FS / BASE_FREQ)  # 单个工频周期的采样点数：30000/60 = 500 点

# -------------------- 信号预处理函数 --------------------
def align_cycle_to_voltage_zero(voltage_cycle, current_cycle):
    """
    将单个周期的电压/电流波形对齐到【电压正过零点】
    作用：确保所有 V-I 轨迹的起始点一致，消除相位随机性
    :param voltage_cycle: 单周期电压数组
    :param current_cycle: 单周期电流数组
    :return: 对齐后的 (voltage_cycle, current_cycle)
    """
    # 寻找第一个“电压从负变正穿过 0”的点（正过零点）
    zero_cross = np.where((voltage_cycle[:-1] <= 0) & (voltage_cycle[1:] > 0))[0]
    
    # 如果没找到正过零点，退而求其次找负过零点
    if len(zero_cross) == 0:
        zero_cross = np.where((voltage_cycle[:-1] >= 0) & (voltage_cycle[1:] < 0))[0]
    
    # 如果还是没找到（异常数据），返回原波形并打印警告
    if len(zero_cross) == 0:
        print("Warning: No zero crossing found, using original cycle.")
        return voltage_cycle, current_cycle
    
    # 循环移位，将过零点移到数组起始位置
    shift = zero_cross[0]
    aligned_v = np.roll(voltage_cycle, -shift)
    aligned_i = np.roll(current_cycle, -shift)
    return aligned_v, aligned_i

def normalize_data(column_data):
    """
    将数据归一化到 [0, 1] 范围
    作用：消除不同电器电压/电流幅值差异，便于 V-I 轨迹绘图
    :param column_data: 原始一维数组
    :return: 归一化后的数组
    """
    return (column_data - column_data.min()) / (column_data.max() - column_data.min())

# -------------------- 电气特征计算函数 --------------------
def compute_power(voltage, current):
    """计算瞬时功率（简单相乘）"""
    return voltage * current

def compute_power_factor(voltage, current):
    """
    计算【功率因数 (PF)】
    公式：PF = 有功功率 P / 视在功率 S
    """
    P_active = np.mean(voltage * current)  # 有功功率：瞬时功率的平均值
    V_rms = np.sqrt(np.mean(voltage**2))   # 电压有效值
    I_rms = np.sqrt(np.mean(current**2))   # 电流有效值
    S_apparent = V_rms * I_rms             # 视在功率
    PF = P_active / S_apparent
    return PF

def compute_hue(voltage, current):
    """
    计算【色调 (Hue)】—— 用于 HSV 颜色编码
    原理：根据 V-I 轨迹上相邻点的“运动方向”计算角度，映射为颜色
    :return: 归一化到 [0, 1] 的色调数组
    """
    H = np.zeros_like(voltage)
    # 遍历每个点，计算与下一个点的相对运动方向
    for j in range(len(voltage) - 1):
        delta_v = voltage[j + 1] - voltage[j]
        delta_i = current[j + 1] - current[j]
        angle = np.arctan2(delta_v, delta_i)  # 计算方向角（弧度，范围 [-π, π]）
        normalized_angle = (angle + np.pi) / (2 * np.pi)  # 归一化到 [0, 1]
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
    # sample_rate = float(sample_rate)
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


# -------------------- 绘图核心函数 --------------------
def plot_3d_power_hue_vi_trajectory(voltage, current, save_path, sample_rate):
    """
    绘制纯净的3D彩色V-I轨迹图（无网格、无坐标轴、白色背景）
    """
    # 时间轴（一个周期16.65ms）
    time_stamps = np.linspace(0, 16.65, len(voltage))
    
    # 计算HSV三通道
    PF = compute_power_factor(voltage, current)
    H = compute_hue(voltage, current)
    S = np.ones_like(H) * (0.5 + 0.5 * PF)
    V_value = compute_third_harmonic(voltage, sample_rate) / (np.abs(voltage).max() + 1e-6)
    V = np.full_like(H, np.clip(V_value, 0, 1))
    
    HSV = np.stack((H, S, V), axis=-1)
    RGB = colors.hsv_to_rgb(HSV)
    RGB = np.clip(RGB, 0, 1)
    
   # 创建 3D 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 逐段绘制轨迹
    for i in range(1, len(voltage)):
        ax.plot(voltage[i-1:i+1], current[i-1:i+1], time_stamps[i-1:i+1],
                color=RGB[i], linewidth=2.5, alpha=0.9)
    
    # 隐藏所有坐标轴元素
    ax.set_axis_off()                     # 完全关闭坐标轴
    ax.grid(False)                        # 关闭网格
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))   # 透明化面板
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo['grid']['linewidth'] = 0        # 隐藏网格线
    ax.yaxis._axinfo['grid']['linewidth'] = 0
    ax.zaxis._axinfo['grid']['linewidth'] = 0
    
    # 设置白色背景（实际上面板已透明，figure背景白色）
    fig.patch.set_facecolor('white')
  
    # 保存图像，去除所有白边
    plt.savefig(save_path)
    plt.close()

# # -------------------- 辅助与并行处理函数 --------------------
# def sort_key(filename):
#     """
#     文件名排序函数：提取文件名中的数字部分进行排序
#     例如：'1.csv' -> 1, '10.csv' -> 10，避免字符串排序的 '10' < '2' 问题
#     """
#     numbers = [int(s) for s in filename.split('.') if s.isdigit()]
#     return numbers[0] if numbers else -1

# def process_one_item(args):
#     try:
#         key, value, base_save_folder_path, sample_rate, FS, BASE_FREQ, CYCLES_FOR_AVG = args
        
#         # 强制类型转换（必须放在所有使用之前）
#         FS = int(FS)
#         BASE_FREQ = int(BASE_FREQ)
#         CYCLES_FOR_AVG = int(CYCLES_FOR_AVG)
#         sample_rate = int(sample_rate)
        
#         current_raw = value['current']
#         voltage_raw = value['voltage']
#         indices = value['indices']
#         file_id = key.split('_')[-1]
        
#         if indices is None:
#             print(f"设备 {file_id} 未找到稳态区间")
#             return (False, file_id, key, "No steady interval")
        
#         steady_start, steady_end = indices
#         print(f"设备 {file_id} 稳态区间采样点: [{steady_start}, {steady_end})")
#         # 在函数外部打开文件，或每次追加
#         with open('steady_intervals.csv', 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([file_id, steady_start, steady_end])
#         # 创建按编号保存的文件夹
#         save_folder_path = os.path.join(base_save_folder_path, file_id)
#         os.makedirs(save_folder_path, exist_ok=True)
        
#         window_len = FS // BASE_FREQ   # 500
#         max_possible_cycles = (steady_end - steady_start) // window_len
#         if max_possible_cycles < CYCLES_FOR_AVG:
#             print(f"Warning: {key} 稳态区间内只有 {max_possible_cycles} 个周期，使用全部可用周期")
#             N_cycles = max_possible_cycles
#         else:
#             N_cycles = CYCLES_FOR_AVG
        
#         if N_cycles == 0:
#             print(f"设备 {file_id} 无完整周期可提取")
#             return (False, file_id, key, "No complete cycles")
        
#         cycles_v, cycles_i = [], []
#         for n in range(N_cycles):
#             start = steady_start + n * window_len
#             end = start + window_len
#             v_cycle = voltage_raw[start:end].copy()
#             i_cycle = current_raw[start:end].copy()
#             v_align, i_align = align_cycle_to_voltage_zero(v_cycle, i_cycle)
#             cycles_v.append(v_align)
#             cycles_i.append(i_align)
        
#         for n, (v_cycle, i_cycle) in enumerate(zip(cycles_v, cycles_i)):
#             v_norm = normalize_data(v_cycle)
#             i_norm = normalize_data(i_cycle)
#             save_path = os.path.join(save_folder_path, f"cycle_{n+1}.png")   # 简化文件名
#             plot_3d_power_hue_vi_trajectory(v_norm, i_norm, save_path,sample_rate)
        
#         return (True, file_id, key, len(cycles_v))
    
#     except Exception as e:
#         import traceback
#         traceback.print_exc()   # 打印详细错误堆栈，便于调试
#         return (False, file_id, key, str(e))
    
# def process_items_parallel(items, base_save_folder_path, sample_rate, FS, BASE_FREQ, CYCLES_FOR_AVG, max_workers=None):
#     """
#     【并行处理主函数】使用多进程加速处理整个 PLAID 数据集
#     """
#     if max_workers is None:
#         max_workers = min(mp.cpu_count(), 16)  # 默认最大进程数不超过 CPU 核数或 16

#     # 构建参数列表：把所有参数打包成元组，传给 process_one_item
#     args_list = [(key, value, base_save_folder_path, sample_rate, FS, BASE_FREQ, CYCLES_FOR_AVG) 
#                  for key, value in items]

#     total = len(args_list)
#     successful = 0
#     failed = 0
#     total_cycles = 0
#     start_time = time.time()

#     # 创建进程池并提交任务
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         future_to_args = {executor.submit(process_one_item, args): args for args in args_list}
#         completed = 0
#         # 逐个获取完成的任务结果
#         for future in as_completed(future_to_args):
#             args = future_to_args[future]
#             file_id, filename = args[0], args[1]
#             try:
#                 result = future.result()
#                 if result[0]:
#                     successful += 1
#                     total_cycles += result[3]
#                     print(f"✓ {filename} 处理成功，生成 {result[3]} 个周期图像")
#                 else:
#                     failed += 1
#                     print(f"✗ {filename} 处理失败: {result[3]}")
#             except Exception as e:
#                 failed += 1
#                 print(f"✗ {filename} 发生异常: {e}")

#             # 每处理完 10 个样本打印一次进度
#             completed += 1
#             if completed % 10 == 0:
#                 elapsed = time.time() - start_time
#                 print(f"进度: {completed}/{total} ({completed/total*100:.1f}%) | "
#                       f"耗时: {elapsed:.1f}s | 成功: {successful} | 失败: {failed}")

#     # 打印最终统计信息
#     elapsed = time.time() - start_time
#     print(f"\n并行处理完成，总耗时: {elapsed:.2f}s，成功: {successful}，失败: {failed}，总周期图像数: {total_cycles}")
#     return successful, failed, total_cycles

# # -------------------- 程序入口 --------------------
# if __name__ == "__main__":
#     import matplotlib
#     matplotlib.use('Agg')
    
#     # 路径配置
#     csv_folder_path = r'/data/users/xzq/nilm/plaid 2014/2014'
#     base_save_folder_path = r'/data/users/xzq/nilm/data'
#     metadata_file = '/data/users/xzq/nilm/plaid 2014/meta_2014.json'
#     os.makedirs(base_save_folder_path, exist_ok=True)
    
#     sample_rate = 30000
#     FS = 30000
#     BASE_FREQ = 60
#     CYCLES_FOR_AVG = 15
    
#     # 加载元数据和信号字典
#     appliance_dict = metadata_submetered(metadata_file)
#     signal_dict = steady_samples_submetered(csv_folder_path + '/', appliance_dict, sample_cycles=CYCLES_FOR_AVG)
#     items = list(signal_dict.items())
#     print(f"共处理 {len(items)} 个电器实例")
    
#     successful = 0
#     failed = 0
#     total_cycles = 0
#     start_time = time.time()
    
#     # 单进程顺序处理
#     for idx, (key, value) in enumerate(items, 1):
#         print(f"处理进度: {idx}/{len(items)}")
#         # 直接调用 process_one_item（注意需要传入所有参数，但 process_one_item 原本设计接收 args 元组）
#         # 为简化，我们直接在这里展开调用，或者复用 process_one_item 函数但改为直接传参
#         # 由于 process_one_item 内部会解包 args，我们构造一个元组即可
#         args = (key, value, base_save_folder_path, sample_rate, FS, BASE_FREQ, CYCLES_FOR_AVG)
#         result = process_one_item(args)
#         if result[0]:
#             successful += 1
#             total_cycles += result[3]
#             print(f"✓ {result[1]} 处理成功，生成 {result[3]} 个周期图像")
#         else:
#             failed += 1
#             print(f"✗ {result[1]} 处理失败: {result[3]}")
        
#         # 每10个打印一次统计
#         if idx % 10 == 0:
#             elapsed = time.time() - start_time
#             print(f"进度: {idx}/{len(items)} ({idx/len(items)*100:.1f}%) | "
#                   f"耗时: {elapsed:.1f}s | 成功: {successful} | 失败: {failed}")
    
#     elapsed = time.time() - start_time
#     print(f"\n处理完成，总耗时: {elapsed:.2f}s，成功: {successful}，失败: {failed}，总周期图像数: {total_cycles}")
# -------------------- 辅助函数 --------------------
def load_valid_ids(csv_path):
    """加载 valid_steady_ids.csv，返回设备ID的集合（整数）"""
    valid_ids = set()
    if not os.path.exists(csv_path):
        print(f"警告: {csv_path} 不存在，将使用空集合（所有设备都采用末尾周期）")
        return valid_ids
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        for row in reader:
            if row:
                try:
                    valid_ids.add(int(row[0]))
                except ValueError:
                    print(f"跳过无效行: {row}")
    print(f"加载了 {len(valid_ids)} 个有效设备ID")
    return valid_ids

def get_last_n_cycles(current, voltage, cycles=CYCLES_FOR_AVG):
    """获取文件末尾的 cycles 个完整周期，返回 (start_sample, end_sample)"""
    total_samples = len(current)
    total_cycles = total_samples // WINDOW_LEN
    if total_cycles == 0:
        return None
    if total_cycles >= cycles:
        start = (total_cycles - cycles) * WINDOW_LEN
        end = total_cycles * WINDOW_LEN
    else:
        start = 0
        end = total_cycles * WINDOW_LEN
    return start, end

# -------------------- 修改后的 process_one_item --------------------
def process_one_item(args):
    try:
        key, value, base_save_folder_path, sample_rate, FS, BASE_FREQ, CYCLES_FOR_AVG, valid_ids_set = args
        
        FS = int(FS); BASE_FREQ = int(BASE_FREQ); CYCLES_FOR_AVG = int(CYCLES_FOR_AVG); sample_rate = int(sample_rate)
        
        current_raw = value['current']
        voltage_raw = value['voltage']
        file_id = key.split('_')[-1]
        file_id_int = int(file_id)
        
        # 判断是否使用原方法
        use_original = (file_id_int in valid_ids_set)
        
        if use_original:
            # 使用原有的 steady_samples_submetered 提供的 indices
            indices = value.get('indices')
            if indices is None:
                print(f"设备 {file_id} 在有效列表中但未找到稳态区间，回退到末尾周期")
                use_original = False
            else:
                steady_start, steady_end = indices
                print(f"设备 {file_id} (原方法) 稳态区间采样点: [{steady_start}, {steady_end})")
                # 写入日志（可选）
                with open('steady_intervals_original.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([file_id, steady_start, steady_end])
        
        if not use_original:
            # 使用末尾15个周期
            result = get_last_n_cycles(current_raw, voltage_raw, CYCLES_FOR_AVG)
            if result is None:
                print(f"设备 {file_id} 文件长度不足一个周期，跳过")
                return (False, file_id, key, "No complete cycles")
            steady_start, steady_end = result
            print(f"设备 {file_id} (末尾周期) 区间采样点: [{steady_start}, {steady_end})")
            with open('steady_intervals_fallback.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([file_id, steady_start, steady_end])
        
        # 创建按编号保存的文件夹
        save_folder_path = os.path.join(base_save_folder_path, file_id)
        os.makedirs(save_folder_path, exist_ok=True)
        
        window_len = FS // BASE_FREQ   # 500
        max_possible_cycles = (steady_end - steady_start) // window_len
        if max_possible_cycles < CYCLES_FOR_AVG:
            print(f"Warning: {key} 区间内只有 {max_possible_cycles} 个周期，使用全部可用周期")
            N_cycles = max_possible_cycles
        else:
            N_cycles = CYCLES_FOR_AVG
        
        if N_cycles == 0:
            print(f"设备 {file_id} 无完整周期可提取")
            return (False, file_id, key, "No complete cycles")
        
        cycles_v, cycles_i = [], []
        for n in range(N_cycles):
            start = steady_start + n * window_len
            end = start + window_len
            v_cycle = voltage_raw[start:end].copy()
            i_cycle = current_raw[start:end].copy()
            v_align, i_align = align_cycle_to_voltage_zero(v_cycle, i_cycle)
            cycles_v.append(v_align)
            cycles_i.append(i_align)
        
        for n, (v_cycle, i_cycle) in enumerate(zip(cycles_v, cycles_i)):
            v_norm = normalize_data(v_cycle)
            i_norm = normalize_data(i_cycle)
            save_path = os.path.join(save_folder_path, f"cycle_{n+1}.png")
            plot_3d_power_hue_vi_trajectory(v_norm, i_norm, save_path, sample_rate)
        
        return (True, file_id, key, len(cycles_v))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (False, file_id, key, str(e))

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    # 路径配置
    csv_folder_path = r'/data/users/xzq/nilm/plaid 2014/2014'
    base_save_folder_path = r'/data/users/xzq/nilm/data_hybrid'  # 新输出目录，避免覆盖
    metadata_file = '/data/users/xzq/nilm/plaid 2014/meta_2014.json'
    valid_csv = 'valid_steady_ids.csv'   # 存放有效设备ID的CSV文件
    os.makedirs(base_save_folder_path, exist_ok=True)
    
    sample_rate = 30000
    FS = 30000
    BASE_FREQ = 60
    CYCLES_FOR_AVG = 15
    
    # 加载有效设备ID集合
    valid_ids_set = load_valid_ids(valid_csv)
    
    # 加载元数据和信号字典（注意：这里仍然会调用 steady_samples_submetered，它内部会计算 indices）
    appliance_dict = metadata_submetered(metadata_file)
    signal_dict = steady_samples_submetered(csv_folder_path + '/', appliance_dict, sample_cycles=CYCLES_FOR_AVG)
    items = list(signal_dict.items())
    print(f"共处理 {len(items)} 个电器实例")
    
    successful = 0
    failed = 0
    total_cycles = 0
    start_time = time.time()
    
    # 单进程顺序处理（避免多进程参数传递复杂）
    for idx, (key, value) in enumerate(items, 1):
        print(f"处理进度: {idx}/{len(items)}")
        # 构造参数元组，多传递一个 valid_ids_set
        args = (key, value, base_save_folder_path, sample_rate, FS, BASE_FREQ, CYCLES_FOR_AVG, valid_ids_set)
        result = process_one_item(args)
        if result[0]:
            successful += 1
            total_cycles += result[3]
            print(f"✓ {result[1]} 处理成功，生成 {result[3]} 个周期图像")
        else:
            failed += 1
            print(f"✗ {result[1]} 处理失败: {result[3]}")
        
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"进度: {idx}/{len(items)} ({idx/len(items)*100:.1f}%) | "
                  f"耗时: {elapsed:.1f}s | 成功: {successful} | 失败: {failed}")
    
    elapsed = time.time() - start_time
    print(f"\n处理完成，总耗时: {elapsed:.2f}s，成功: {successful}，失败: {failed}，总周期图像数: {total_cycles}")
import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from functools import partial

# 保持原代码的计算函数完全不变
def normalize_data(column_data):
    return (column_data - column_data.min()) / (column_data.max() - column_data.min())

def compute_power(voltage, current):
    return voltage * current

def compute_power_factor(voltage, current):
    P_active = np.mean(voltage * current)
    V_rms = np.sqrt(np.mean(voltage**2))
    I_rms = np.sqrt(np.mean(current**2))
    S_apparent = V_rms * I_rms
    PF = P_active / S_apparent
    return PF

def compute_hue(voltage, current):
    # 保持原代码的循环计算，确保色彩一致
    H = np.zeros_like(voltage)
    for j in range(len(voltage) - 1):
        delta_v = voltage[j + 1] - voltage[j]
        delta_i = current[j + 1] - current[j]
        angle = np.arctan2(delta_v, delta_i)
        normalized_angle = (angle + np.pi) / (2 * np.pi)
        H[j] = normalized_angle
    H[-1] = H[-2]
    return H

def compute_third_harmonic(data, sample_rate):
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)
    third_harmonic_freq = 3 * 60
    third_harmonic_index = np.argmin(np.abs(freqs - third_harmonic_freq))
    third_harmonic_magnitude = np.abs(fft_result[third_harmonic_index])
    return third_harmonic_magnitude



def plot_3d_power_hue_vi_trajectory(data, save_path, sample_rate):
    voltage = data.iloc[:, 1].values
    current = data.iloc[:, 0].values
    time_interval = 16.65 / len(voltage)
    time_stamps = np.arange(0, 16.65, time_interval)

    PF = compute_power_factor(voltage, current)
    H = compute_hue(voltage, current)
    S = np.ones_like(H) * (0.5 + 0.5 * PF)
    third_harmonic = compute_third_harmonic(voltage, sample_rate)
    if third_harmonic > 0:
        V_value = third_harmonic / np.max(third_harmonic)
    else:
        V_value = 0.5
    V = np.full_like(H, V_value)

    HSV = np.stack((H, S, V), axis=-1)
    RGB = colors.hsv_to_rgb(HSV)

    # 固定图像尺寸为224x224像素
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 增加轨迹线宽
    line_width = 3.0
    alpha = 0.9
    
    # 绘制轨迹
    for i in range(1, len(voltage)):
        ax.plot(voltage[i-1:i+1], current[i-1:i+1], time_stamps[i-1:i+1], 
                color=RGB[i], linewidth=line_width, alpha=alpha)
    
    # 设置坐标轴范围，让轨迹尽可能占据空间
    # 计算数据范围并稍微扩展一点，确保轨迹不会紧贴边界
    v_min, v_max = voltage.min(), voltage.max()
    i_min, i_max = current.min(), current.max()
    t_min, t_max = 0, 16.65
    
    # 扩展5%的范围，让轨迹在边界内
    v_range = v_max - v_min
    i_range = i_max - i_min
    v_min_expanded = v_min - 0.05 * v_range
    v_max_expanded = v_max + 0.05 * v_range
    i_min_expanded = i_min - 0.05 * i_range
    i_max_expanded = i_max + 0.05 * i_range
    
    ax.set_xlim([v_min_expanded, v_max_expanded])
    ax.set_ylim([i_min_expanded, i_max_expanded])
    ax.set_zlim([t_min, t_max])
    
    # 添加细小的网格
    # 网格颜色设置为浅灰色，线宽很细
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, color='gray')
    
    # 隐藏坐标轴标签和刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    
    # 设置背景为白色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    # 隐藏坐标轴线
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # 保存图像，去除所有白边
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none', dpi=100)
    plt.close()



def sort_key(filename):
    numbers = [int(s) for s in filename.split('.') if s.isdigit()]
    return numbers[0] if numbers else -1
    
def process_file_batch(args):
    """处理单个文件的函数，批量处理版本"""
    file_id, filename, csv_folder_path, base_save_folder_path, sample_rate = args
    
    try:
        # 创建保存文件夹
        save_folder_path = os.path.join(base_save_folder_path, str(file_id))
        os.makedirs(save_folder_path, exist_ok=True)
        
        # 读取CSV文件
        csv_file_path = os.path.join(csv_folder_path, filename)
        data = pd.read_csv(csv_file_path, header=None)

        # 归一化数据
        data.iloc[:, 0] = normalize_data(data.iloc[:, 0])
        data.iloc[:, 1] = normalize_data(data.iloc[:, 1])

        samples_per_cycle = 500
        number_of_cycles = len(data) // samples_per_cycle
        
        # 预计算一些重复使用的值
        sample_rate_local = sample_rate
        
        # 批量处理所有周期
        for i in range(number_of_cycles):
            start = i * samples_per_cycle
            end = (i + 1) * samples_per_cycle
            cycle_data = data.iloc[start:end, :]
            save_path = os.path.join(save_folder_path, f"VI Trajectory {i + 1}.png")
            plot_3d_power_hue_vi_trajectory(cycle_data, save_path, sample_rate_local)

        return True, file_id, filename
        
    except Exception as e:
        return False, file_id, f"{filename}: {str(e)}"

def process_files_in_parallel(args_list, max_workers=None):
    """并行处理文件的主函数"""
    if max_workers is None:
        # 自动确定进程数：CPU核心数 * 2
        max_workers = min(mp.cpu_count() * 2, 32, len(args_list))
    
    print(f"使用 {max_workers} 个进程并行处理")
    
    start_time = time.time()
    successful = 0
    failed = 0
    processed_files = []
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_file_batch, args): args 
                         for args in args_list}
        
        # 收集结果并显示进度
        completed = 0
        total = len(args_list)
        
        for future in as_completed(future_to_file):
            # 获取文件信息
            args = future_to_file[future]
            file_id, filename = args[0], args[1]
            
            try:
                success, processed_id, result = future.result()
                if success:
                    successful += 1
                    processed_files.append(processed_id)
                    print(f"✓ 成功处理: {filename}")
                else:
                    failed += 1
                    print(f"✗ 处理失败: {result}")
                    
                completed += 1
                
                # 显示进度（每10个文件显示一次，避免过多输出）
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / max(1, completed)
                    remaining = avg_time * (total - completed)
                    
                    print(f"进度: {completed}/{total} ({completed/total*100:.1f}%) | "
                          f"已用: {elapsed:.1f}s | 剩余: {remaining:.1f}s | "
                          f"速度: {completed/elapsed:.2f} 文件/秒")
                      
            except Exception as e:
                failed += 1
                print(f"✗ 处理异常 {filename}: {str(e)}")
                completed += 1
    
    elapsed_time = time.time() - start_time
    return successful, failed, elapsed_time, processed_files

def main():
    # 配置路径
    csv_folder_path = r'/data/users/xzq/nilm/plaid 2014/2014'
    base_save_folder_path = r'/data/users/xzq/nilm/Colorful VI Trajectory_check'
    progress_file = os.path.join(base_save_folder_path, 'progress.txt')
    sample_rate = 30000
    start_id = 176
    
    # 确保保存目录存在
    os.makedirs(base_save_folder_path, exist_ok=True)
    
    def get_last_processed_id():
        """获取上次处理的最后ID"""
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return int(f.read().strip())
            except:
                return 0
        return 0
    
    # 检查进度文件
    last_processed = get_last_processed_id()
    actual_start = max(start_id, last_processed + 1)
    print(f"上次处理到ID: {last_processed}")
    print(f"计划从ID: {start_id} 开始")
    print(f"实际从ID: {actual_start} 开始")
    
    # 获取所有CSV文件
    all_csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    
    # 提取ID并排序
    csv_files_with_ids = []
    for filename in all_csv_files:
        try:
            # 从文件名提取数字ID
            base_name = os.path.splitext(filename)[0]
            file_id = int(''.join(filter(str.isdigit, base_name)))
            csv_files_with_ids.append((file_id, filename))
        except:
            continue
    
    csv_files_with_ids.sort(key=lambda x: x[0])
    
    # 筛选从起始ID开始的文件
    files_to_process = [(file_id, filename) for file_id, filename in csv_files_with_ids 
                       if file_id >= actual_start]
    
    print(f"总共发现 {len(all_csv_files)} 个CSV文件")
    print(f"需要处理 {len(files_to_process)} 个文件（从ID={actual_start}开始）")
    
    if not files_to_process:
        print("没有需要处理的文件")
        return
    
    # 准备参数列表
    args_list = [(file_id, filename, csv_folder_path, base_save_folder_path, sample_rate) 
                 for file_id, filename in files_to_process]
    
    # 并行处理文件
    successful, failed, elapsed_time, processed_files = process_files_in_parallel(args_list)
    
    # 更新进度文件（只记录最大ID）
    if processed_files:
        max_id = max(processed_files)
        with open(progress_file, 'w') as f:
            f.write(str(max_id))
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"成功: {successful}, 失败: {failed}")
    print(f"总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    if successful > 0:
        print(f"平均每个文件: {elapsed_time/successful:.2f}秒")
        print(f"处理速度: {successful/elapsed_time:.2f} 文件/秒")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 设置matplotlib为非交互式后端
    import matplotlib
    matplotlib.use('Agg')  # 不显示图形界面
    
    # 设置matplotlib默认参数，防止中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置matplotlib缓存，提高性能
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['figure.dpi'] = 100
    
    main()
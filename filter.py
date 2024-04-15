import numpy as np
from scipy.signal import medfilt
from filterpy.kalman import KalmanFilter

def clean_rssi_values(rssi_values_sample):
    rssi_list = rssi_values_sample.copy()

    # 一次性移除异常值
    while True:
        mean_rssi = np.mean(rssi_list)
        max_rssi = max(rssi_list)
        min_rssi = min(rssi_list)

        # 记录要移除的异常值索引
        indices_to_remove = []
        for i, rssi in enumerate(rssi_list):
            if (max_rssi - mean_rssi) / mean_rssi > 0.2 and rssi == max_rssi:
                indices_to_remove.append(i)
            elif (mean_rssi - min_rssi) / mean_rssi > 0.2 and rssi == min_rssi:
                indices_to_remove.append(i)
        # 移除异常值
        if indices_to_remove:
            rssi_list = [rssi for i, rssi in enumerate(rssi_list) if i not in indices_to_remove]
        else:
            break

    return rssi_list


def gaussian_filter(data,window_size):
    filtered_data = []
    for i in range(len(data)):
        if i < window_size:
            window = data[:i+1]
        else:
            window = data[i-window_size+1:i+1]

        mean_value = np.mean(window)
        std_dev = np.std(window)
  
        # 选择在（u - a，u + a）区间的数据
        filtered_values = [x for x in window if mean_value - std_dev <= x <= mean_value + std_dev]
        filtered_values = np.mean(filtered_values)
        filtered_data.append(filtered_values)
    return filtered_data

def moving_average_filter(data, window_size):
    filtered_data = []
    for i in range(len(data)):
        if i < window_size:
            window = data[:i+1]
        else:
            window = data[i-window_size+1:i+1]
        filtered_value = sum(window) / len(window)
        filtered_data.append(filtered_value)
    return filtered_data

def kalman_filter(data, process_variance, measurement_variance):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([data[0]])  # 初始状态估计
    kf.P *= 10  # 初始状态协方差估计
    
    kf.F = np.array([[1]])  # 状态转移矩阵
    kf.H = np.array([[1]])  # 观测矩阵
    kf.R *= measurement_variance  # 观测噪声协方差矩阵
    kf.Q *= process_variance  # 过程噪声协方差矩阵

    filtered_data = []
    for measurement in data:
        kf.predict()  # 预测下一时刻状态
        kf.update(measurement)  # 更新状态估计
        filtered_data.append(kf.x[0])
    
    return filtered_data

def alpha_beta_gama(z):
    initial_guess = np.mean(z)
    number_of_values = len(z)
    x = [0] * number_of_values
    x[0] = initial_guess
    for n in range(1, number_of_values):
        if n >= 10:
            kn = 1/10
        else:
            kn = 1/n
        # z[n-1] is z[n] here
        x[n] = x[n-1] + ( kn * (z[n-1] - x[n-1]))
    return x

def median_filter(data, window_size):
    return medfilt(data, kernel_size=window_size)

def weighted_blend_filter(data, window_size, process_variance, measurement_variance, alpha_beta_gama_weight=0.25, moving_average_weight=0.25, kalman_weight=0.25, gaussian_weight=0.25):
    # Apply each filter
    alpha_beta_gama_output = alpha_beta_gama(data)
    moving_average_output = moving_average_filter(data, window_size)
    kalman_output = kalman_filter(data, process_variance, measurement_variance)
    gaussian_output = gaussian_filter(data, window_size)
    # Apply weights and blend the filtered outputs
    blended_output = []
    for i in range(len(data)):
        blended_value = (alpha_beta_gama_output[i] * alpha_beta_gama_weight + 
                         moving_average_output[i] * moving_average_weight + 
                         kalman_output[i] * kalman_weight + 
                         gaussian_output[i] * gaussian_weight)
        blended_output.append(blended_value)
    
    return blended_output

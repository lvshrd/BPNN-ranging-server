import os
import re
import filter
from scipy.optimize import curve_fit
# 定义拟合函数
def curve_func(rssi_value, A, n):
    return 10 ** ((A - rssi_value) / (10 * n))

data_path_base = 'D:/311/FYP/my_project/Linear Regression/data_env_'
envs = [0, 1, 2, 3]  # 设置环境数量
for env in envs:
    data_path = data_path_base + str(env)
    if os.path.exists(data_path):
        # file_pattern = re.compile(r'RSSI-([\d.]+)m\.txt')
        file_pattern = re.compile(r"RSSI-(\d+(\.\d+)?)m\.txt")
        file_names = os.listdir(data_path)
        sorted_file_names = sorted(file_names, key=lambda x: float(file_pattern.match(x).group(1)))

        env_rssi_values = []
        env_distances = []

        for file_name in sorted_file_names:
            match = file_pattern.match(file_name)
            if match:
                distance = float(match.group(1))
                file_path = os.path.join(data_path, file_name)

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                rssi_values = []
                distances = []
                for line in lines:
                    rssi_match = re.search(r'\| (-?\d+)$', line)
                    if rssi_match:
                        rssi = int(rssi_match.group(1))
                        rssi_values.append(rssi)
                        distances.append(distance)

                cleaned_rssi_values = filter.clean_rssi_values(rssi_values)
                env_rssi_values.extend(cleaned_rssi_values)
                env_distances.extend(distances)

        filtered_env_rssi_values = filter.weighted_blend_filter(env_rssi_values, window_size=51, process_variance=1, measurement_variance=1000)
        params, params_covariance = curve_fit(curve_func, filtered_env_rssi_values, env_distances, p0=[-50, 1])
        A, n = params
        print("Estimated parameters:")
        print("A:", A)
        print("n:", n)
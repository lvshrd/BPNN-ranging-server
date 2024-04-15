import numpy as np
import os

def shadowing_model(distance, reference_distance=1, reference_rssi=-50, path_loss_exponent=2, shadowing_std=3, num_samples=500):
    path_loss = reference_rssi - 10 * path_loss_exponent * np.log10(distance / reference_distance)
    shadowing = np.random.normal(scale=shadowing_std, size=num_samples)
    return path_loss + shadowing

# 参数设置
num_files = 10  # 生成文件数量
min_distance = 1  # 最小距离
max_distance = 10  # 最大距离
reference_distance = 1  # 参考距离
reference_rssi = -50  # 在参考距离处的 RSSI
path_loss_exponent = 2  # 路径损耗指数
shadowing_std = 3  # 随机阴影衰减的标准差
num_samples = 500  # 每个文件中的样本数

# 创建目录保存数据文件
directory_name = f"data_env{int(path_loss_exponent)}"
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# 生成数据文件
for i in range(num_files):
    distance = np.random.uniform(min_distance, max_distance)
    rssi_data = shadowing_model(distance, reference_distance, reference_rssi, path_loss_exponent, shadowing_std, num_samples)
    file_path = directory_name +f"/distance_{distance:.2f}_m.txt"
    with open(file_path, 'w') as f:
        for rssi in rssi_data:
            f.write(f"{rssi:.2f}\n")

print("数据文件已生成。")

import os

# 创建新文件夹
output_folder = 'D:/311/FYP/my_project/Linear Regression/data_env_0'
os.makedirs(output_folder, exist_ok=True)

# 遍历原始数据文件
for i in range(1, 11):
    input_filename = f'D:/311/FYP/my_project/Linear Regression/test/data_env_0/RSSI-{i}m.txt'
    output_filename = os.path.join(output_folder, f'RSSI-{i}m.txt')
    
    # 读取前800行数据
    with open(input_filename, 'r') as f:
        lines = f.readlines()[:480]
    
    # 将数据写入新文件
    with open(output_filename, 'w') as f:
        f.writelines(lines)

print("数据对齐完成，已存储到 data_env1_aligned 文件夹中。")

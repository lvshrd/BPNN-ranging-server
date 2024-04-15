import os

# 定义输入和输出文件夹
input_folder = "RSSI-wall"
output_folder = "data_env_0"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的每个文件
for filename in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, filename)
    output_file_path = os.path.join(output_folder, filename)
    
    # 打开输入文件
    with open(input_file_path, 'r') as input_file:
        # 创建一个列表，用于存储符合条件的行
        filtered_lines = []
        # 遍历输入文件的每一行
        for line in input_file:
            # 检查是否包含 "ESP_GATTS_DEMO"
            if "| ESP_GATTS_DEMO |" in line:
                # 如果包含，则保留该行
                filtered_lines.append(line)
        
        # 打开输出文件，写入符合条件的行
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(filtered_lines)

print("Filtering complete.")

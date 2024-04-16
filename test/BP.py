import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import filter

# 定义神经网络模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out) 
        return out
    
 
# 数据路径
data_path_base = 'D:/311/FYP/my_project/Linear Regression/data_env_'
envs = [1, 2, 3]  # 设置环境数量
file_count = 10  # 每个环境的文件数量


# def _filter(rssi_value_list):
#     km = kalman()
#     filtered_rssi_values = []
#     for z in rssi_value_list:
#         filtered_rssi_values.append(km.kalman_filter(z))
#     return filtered_rssi_values

def _showplot(list1,list2,label1,label2,title):
    # print("Type of list2:", type(list2))
    # print("Shape of list2:", np.array(list2).shape)
    # 绘制图像
    plt.plot(list1, label=label1)
    plt.plot(list2, label=label2)
    # 添加标题和标签
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 添加图例
    plt.legend()
    # 显示图像
    plt.show()


# 训练不同节点的模型
for env in envs:
    data_path = data_path_base + str(env)
    model = None  # 每次循环前重置模型
    # 读取数据集文件
    if os.path.exists(data_path):
        file_pattern = re.compile(r'RSSI-([\d.]+)m\.txt')
        file_names = os.listdir(data_path)
        # 对文件名按照数字部分进行排序
        sorted_file_names = sorted(file_names, key=lambda x: int(file_pattern.match(x).group(1)))

        env_rssi_values = []
        env_distances = []

        for file_name in sorted_file_names:
            match = file_pattern.match(file_name)
            if match:
                distance = float(match.group(1))
                file_path = os.path.join(data_path, file_name)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
            # 准备数据
            rssi_values = []
            distances = []
            for line in lines:
                rssi_match = re.search(r'\| (-?\d+)$', line)
                if rssi_match:
                    rssi = int(rssi_match.group(1))
                
                    rssi_values.append(rssi)
                    distances.append(distance)  # 使用文件序号作为距离标签
                    

            cleaned_rssi_values = filter.clean_rssi_values(rssi_values)

            # filtered_rssi_values = filter.weighted_blend_filter(cleaned_rssi_values,window_size=100 , process_variance=50,measurement_variance=500)
            # filtered_rssi_values = filter.moving_average_filter(cleaned_rssi_values,window_size=100 , process_variance=50,measurement_variance=500)
            # _showplot(rssi_values,filtered_rssi_values ,"raw","filtered","distance at"+str(distance))

            env_rssi_values.extend(cleaned_rssi_values)
            env_distances.extend(distances)

        filtered_env_rssi_values = filter.weighted_blend_filter(env_rssi_values,window_size=100 , process_variance=50,measurement_variance=500)
        # _showplot(env_rssi_values,filtered_env_rssi_values ,"raw","filtered","distance at"+str(distance))

        X_train, X_test, y_train, y_test = train_test_split(filtered_env_rssi_values, env_distances, test_size=0.2, random_state=42)

        # 转换为NumPy数组并标准化
        X_train = np.array(X_train).reshape(-1, 1)
        y_train = np.array(y_train).reshape(-1, 1)
        X_test = np.array(X_test).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        X_train_stand = (X_train - np.mean(X_train)) / np.std(X_train)
        X_test_stand = (X_test - np.mean(X_test)) / np.std(X_test)
        # 是否对输出(距离)也进行归一化
        y_train_stand = (y_train - np.mean(y_train)) / np.std(y_train)
        y_test_stand = (y_test - np.mean(y_test)) / np.std(y_test)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.tensor(X_train_stand, dtype=torch.float32) #(1,4000)
        y_train_tensor = torch.tensor(y_train_stand, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_stand, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_stand, dtype=torch.float32)

        input_dim = X_train_tensor.shape[1]
        hidden_dim = 26  # 隐含层大小为 26
        output_dim = 1
        model = MLP(input_dim, hidden_dim, output_dim)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # 训练模型
        num_epochs = 1000  # 可根据需要调整
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # # 测试模型
        # with torch.no_grad():
        #     test_outputs = model(X_test_tensor)
        #     # 反向归一化模型的输出
        #     output_original_scale = test_outputs * np.std(y_train) + np.mean(y_train)

        #     test_outputs_numpy = output_original_scale.detach().numpy()

        #     # 绘制散点图
        #     plt.figure(figsize=(8, 6))
        #     plt.scatter(X_test, y_test, label='True')
        #     plt.scatter(X_test, test_outputs_numpy, label='Predicted')
        #     plt.xlabel('X_test')
        #     plt.ylabel('Output')
        #     plt.title('Regression Fit on Test Data')
        #     plt.legend()
        #     plt.show()

        #     test_loss = criterion(test_outputs, y_test_tensor)
        #     print(f'Test Loss: {test_loss.item():.4f}')

        # 保存模型
        model_path = f'model_env{env}.pth'
        # torch.save(model.state_dict(), model_path)
        torch.save(model, model_path)

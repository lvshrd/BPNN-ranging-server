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
data_path_base = 'D:/311/FYP/my_project/Linear Regression/data_env'
envs = [1, 2, 3]  # 根据你的实际情况设置环境数量
file_count = 10  # 每个环境的文件数量

def alpha_beta_gama(z, initial_guess, number_of_values):
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

def _filter(rssi_value_list):
    # filtered_rssi_values = []
    # for distance in range(number_of_distance):
    initial_guess = np.mean(rssi_value_list)
    # print(f'mean:{initial_guess}')
    # print(f'length:{len(rssi_value_list)}')
    
    z = rssi_value_list
    x = alpha_beta_gama(z, initial_guess, len(rssi_value_list))
        # filtered_rssi_values.append(x)
    return x

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
        file_pattern = re.compile(r'distance_([\d.]+)_m\.txt')
        file_names = os.listdir(data_path)
        env_rssi_values = []
        env_distances = []

        for file_name in file_names:
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
                rssi = float(line.strip())
                rssi_values.append(rssi)
                distances.append(distance)  # 使用文件序号作为距离标签
            
            # filtered_rssi_values = _filter(rssi_values)
            # showplot(env_rssi_values,filtered_rssi_values,'raw','filtered',"distance at "+str(distance)+"m, window_size ="+str(window_size))
            env_rssi_values.extend(rssi_values)
            env_distances.extend(distances)
        process_variance1 = 0.1
        measurement_variance1 = 1000
        process_variance2 = 10
        measurement_variance2 = 100
        filtered_rssi_values1 = filter.weighted_blend_filter(env_rssi_values,101, 1, 1000)
        filtered_rssi_values2 = filter.weighted_blend_filter(env_rssi_values, 51, 1, 1000)
        # # _showplot(env_rssi_values,filtered_rssi_values,'raw','filtered',"window_size ="+str(window_size))
        # plt.plot(env_rssi_values, label='raw')
        # plt.plot(filtered_rssi_values1, label="0.5 kalman+0.25 gaussian+0.2 average+0.05median")
        # plt.plot(filtered_rssi_values2, label="kalman_filter")
        # # 添加标题和标签
        # plt.title('performance of kalman_filter')
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # # 添加图例
        # plt.legend()
        # # 显示图像
        # plt.show()
        # X_train, X_test, y_train, y_test = train_test_split(env_rssi_values, env_distances, test_size=0.2, random_state=42)

        # # 转换为NumPy数组并标准化
        # X_train = np.array(X_train).reshape(-1, 1)
        # y_train = np.array(y_train).reshape(-1, 1)
        # X_test = np.array(X_test).reshape(-1, 1)
        # y_test = np.array(y_test).reshape(-1, 1)
        # X_train_stand = (X_train - np.mean(X_train)) / np.std(X_train)
        # X_test_stand = (X_test - np.mean(X_test)) / np.std(X_test)
        # # 是否对输出(距离)也进行归一化
        # y_train_stand = (y_train - np.mean(y_train)) / np.std(y_train)
        # y_test_stand = (y_test - np.mean(y_test)) / np.std(y_test)
        
        # # 转换为PyTorch张量
        # X_train_tensor = torch.tensor(X_train_stand, dtype=torch.float32) #(1,4000)
        # y_train_tensor = torch.tensor(y_train_stand, dtype=torch.float32)
        # X_test_tensor = torch.tensor(X_test_stand, dtype=torch.float32)
        # y_test_tensor = torch.tensor(y_test_stand, dtype=torch.float32)

        # input_dim = X_train_tensor.shape[1]
        # hidden_dim = 26  # 隐含层大小为 26
        # output_dim = 1
        # model = MLP(input_dim, hidden_dim, output_dim)

        # # 定义损失函数和优化器
        # criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01)

        # # 训练模型
        # num_epochs = 1000  # 可根据需要调整
        # for epoch in range(num_epochs):
        #     # Forward pass
        #     outputs = model(X_train_tensor)
        #     loss = criterion(outputs, y_train_tensor)
            
        #     # Backward pass and optimization
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
            
        #     if (epoch+1) % 100 == 0:
        #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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

        # # 保存模型
        # model_path = f'model_env{env}.pt'
        # torch.save(model.state_dict(), model_path)

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import filter

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

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def preprocess_data(self):
        if os.path.exists(self.data_path):
            file_pattern = re.compile(r'RSSI-([\d.]+)m\.txt')
            file_names = os.listdir(data_path)
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

            filtered_env_rssi_values = filter.weighted_blend_filter(env_rssi_values, window_size=100, process_variance=50, measurement_variance=500)

            X_train, X_test, y_train, y_test = train_test_split(filtered_env_rssi_values, env_distances, test_size=0.2, random_state=42)

            X_train = np.array(X_train).reshape(-1, 1)
            y_train = np.array(y_train).reshape(-1, 1)
            X_test = np.array(X_test).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)
            X_train_stand = (X_train - np.mean(X_train)) / np.std(X_train)
            X_test_stand = (X_test - np.mean(X_test)) / np.std(X_test)
            y_train_stand = (y_train - np.mean(y_train)) / np.std(y_train)
            y_test_stand = (y_test - np.mean(y_test)) / np.std(y_test)

            X_train_tensor = torch.tensor(X_train_stand, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_stand, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_stand, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test_stand, dtype=torch.float32)

            return X_train_tensor, y_train_tensor,X_test_tensor,y_test_tensor, np.mean(y_train), np.std(y_train),X_test, y_test

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, num_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_model(self, X_train_tensor, y_train_tensor):
        for epoch in range(self.num_epochs):
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    # 数据路径
    data_path_base = 'D:/311/FYP/my_project/Linear Regression/data_env_'
    envs = [0, 1, 2, 3]  # 设置环境数量
    model_path_base = 'D:/311/FYP/my_project/Linear Regression/'
    
    # 保存模型
    for env in envs:
        data_path = data_path_base + str(env)
        if os.path.exists(data_path):
            # 创建数据预处理器
            data_preprocessor = DataPreprocessor(data_path)
            # 数据预处理
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, y_train_mean,y_train_std, X_test, y_test= data_preprocessor.preprocess_data()

            # 定义模型参数
            input_dim = 1
            hidden_dim = 26
            output_dim = 1
            num_epochs = 1000

            # 初始化模型
            model = MLP(input_dim, hidden_dim, output_dim)

            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # 创建模型训练器并训练模型
            model_trainer = ModelTrainer(model, criterion, optimizer, num_epochs)
            model_trainer.train_model(X_train_tensor, y_train_tensor)

        
            data_path = data_path_base + str(env)
            if os.path.exists(data_path):
                model_path = os.path.join(model_path_base, f'model_env{env}.pt')
                torch.save(model.state_dict(), model_path)

            # 测试模型
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                
                # 反向归一化模型的输出
                output_original_scale = test_outputs * y_train_std + y_train_mean
                test_outputs_numpy = output_original_scale.detach().numpy()

                y_shadowing = pow(10, (-44- X_test)/18.0)
                # 计算均方根误差
                rmse = np.sqrt(mean_squared_error(y_test, y_shadowing))
                # 计算平均绝对误差
                mae = mean_absolute_error(y_test, y_shadowing)
                print("RMSE Shadowing:", rmse)
                print("MAE Shadowing:", mae)

                # 计算均方根误差
                rmse = np.sqrt(mean_squared_error(y_test, test_outputs_numpy))
                # 计算平均绝对误差
                mae = mean_absolute_error(y_test, test_outputs_numpy)
                print('env'+ str(env) +"RMSE BPmodel:", rmse)
                print('env'+ str(env) +"MAE BPmodel:", mae)
              
                # 绘制散点图
                plt.figure(figsize=(8, 6))
                plt.scatter(X_test, y_test, label='True Value')
                plt.scatter(X_test, y_shadowing, label='Shadowing')
                plt.scatter(X_test, test_outputs_numpy, label='BP Predicted')
                plt.xlabel('RSSI Value')
                plt.ylabel('Distance')
                plt.title('Fit on Test Data in Env '+ str(env))
                plt.legend()
                plt.show()

                test_loss = criterion(test_outputs, y_test_tensor)
                print(f'Test Loss: {test_loss.item():.4f}')
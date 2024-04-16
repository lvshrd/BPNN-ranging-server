import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data.distributed
from BPmodel import MLP, DataPreprocessor
from flask import Flask, request, jsonify

model_path_base = 'D:/311/FYP/my_project/Linear Regression/'
envs = [1, 2, 3]  # 设置环境数量
for env in envs:
    model_path = 'model_env'+ str(env)+'.pth'# 加载模型
    new_model_path = 'model_env'+ str(env)+'.pt'
    if os.path.exists(model_path):
        # 加载模型
        model = MLP(1, 26, 1)
        model.load_state_dict(torch.load(model_path))
        # model = torch.load(model_path)
        model.eval()
        input_shape = (1,1)
        traced_model = torch.jit.trace(model, torch.rand(input_shape))
        torch.jit.save(traced_model, new_model_path)

# 输入一个 RSSI 值进行预测
'''
至于在部署时如何对单个 RSSI 值进行归一化处理，可以保存训练集的均值和标准差，在部署时使用这些统计量对输入的单个 RSSI 值进行归一化处理。
'''

X_test = [-57,-58,-57,-49,-56,-55,-67]
X_test = np.array(X_test).reshape(-1, 1)
X_test_stand = (X_test - np.mean(X_test)) / np.std(X_test)
data_path_base = 'D:/311/FYP/my_project/Linear Regression/data_env_'


with torch.no_grad():
    predicted_distance = model(torch.tensor(X_test_stand, dtype=torch.float32))
    predicted_distance = predicted_distance * 2.7799370586850793 + 5.813296292748348
    test_outputs_numpy = predicted_distance.detach().numpy()
    print("Predicted distance:", test_outputs_numpy)

# # 绘制实际距离与预测距离的散点图
# plt.scatter(actual_distances, predicted_distances)
# plt.xlabel('Actual Distance')
# plt.ylabel('Predicted Distance')
# plt.title('Actual vs. Predicted Distance')
# plt.show()
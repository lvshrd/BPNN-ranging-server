from flask import Flask, request, jsonify
import torch
import numpy as np
from BPmodel import MLP
import filter
import os
import logging

logging.basicConfig(filename='flask_server.log', level=logging.INFO)

app = Flask(__name__)

# 初始化突变检测相关变量
prev_rssi = {}
prev_env = {}
# 定义突变检测阈值
mutation_threshold = 7  
history_max_length = 200
mutation_detection_window = 10
# 获取环境模型，默认使用model_env1
env_model_index = 1 

# 加载预训练模型
model_path_base = 'model/'
envs = [0, 1, 2, 3]  # 设置环境数量
models = {}
for env in envs:
    model_path = model_path_base+'model_env'+ str(env)+'.pt'
    if os.path.exists(model_path):
        # 首先实例化模型类
        model = MLP(input_dim=1, hidden_dim=26, output_dim=1)
        try:
            # 加载模型状态字典
            model.load_state_dict(torch.load(model_path))
            models[env] = model
            print(f"Model for environment {env} loaded successfully.")
        except Exception as e:
            print(f"Failed to load model for environment {env}: {e}")
# 检查加载的模型数量
print(f"Total models loaded: {len(models)}")

# RSSI突变检测函数
def detect_rssi_mutation(rssi_values):
    rssi_array = np.array(rssi_values)
    return np.var(rssi_array) > mutation_threshold
    
# 处理数据并进行预测的函数
def process_data_and_predict(device_name, rssi_values):
    global prev_env
    result = {}
    try:
        result['device_name'] = device_name
        # 清洗和滤波处理RSSI值
        cleaned_rssi = filter.clean_rssi_values(rssi_values)
        filtered_rssi = filter.gaussian_filter(cleaned_rssi,50,500)

        # 转换为数组并进行归一化处理
        rssi_array = np.array(filtered_rssi).reshape(-1,1)
        normalized_rssi = (rssi_array - np.mean(rssi_array)) / np.std(rssi_array)
        env_model = models.get(prev_env[device_name])
        
        if env_model is None:
            raise Exception("Model for environment 1 is not loaded.")
        # 使用模型预测距离值
        rssi_tensor = torch.Tensor(normalized_rssi[-1]).unsqueeze(0).unsqueeze(2)
        distance_prediction = env_model(rssi_tensor).item()
        distance_prediction = distance_prediction * 2.7799370586850793 + 5.813296292748348

        print(f"{device_name} Predicted distance:{distance_prediction}")

        if len(rssi_values) >= mutation_detection_window:
            # 检查RSSI突变情况
            mutation_detected = detect_rssi_mutation(rssi_values[-mutation_detection_window:])
            if mutation_detected:
                # RSSI突变情况，根据情况选择不同的模型进行预测
                prev_env[device_name] = 0 if rssi_values[-2]+ rssi_values[-1]  > rssi_values[-5]+ rssi_values[-4]  else 1
                print(f"{device_name} Predicted distance:{distance_prediction}")

        result['flag'] = prev_env[device_name]
        result['distance_prediction'] = distance_prediction

    except Exception as e:
        result['error'] = str(e)
    return jsonify(result)


def hello():
    return 'hello world'

@app.route('/predict', methods=['POST'])
def predict():
    global prev_rssi, prev_env
    # 从 Android 应用程序发送的数据
    data = request.json  # 假设数据以 JSON 格式发送

    # 获取设备名称和当前 RSSI 值
    device_name = data.get('device_name')
    mac_address = data.get('mac')
    current_rssi = data.get('rssi_value')

    logging.info(f"Received data: device_name={device_name}, mac={mac_address}, rssi_value={current_rssi}")
    # 初始化返回结果
    result = {}

    # 检查是否已经收到过该设备的数据，如果没有，则初始化其对应的 prev_rssi 列表
    if device_name not in prev_rssi:
        prev_rssi[device_name] = []
        prev_env[device_name] = env_model_index

    # 更新该设备对应的 prev_rssi 列表
    prev_rssi[device_name].append(current_rssi)
    if len(prev_rssi[device_name]) > history_max_length:
        prev_rssi[device_name] = prev_rssi[device_name][-history_max_length:]

    # 处理数据并进行预测
    result = process_data_and_predict(device_name, prev_rssi[device_name])
    print(result)

    return result

if __name__ == '__main__':
    app.run(host='172.25.5.223', port=5000,debug=False)
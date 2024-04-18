from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from BPmodel import MLP
import filter
import os
import logging
logging.basicConfig(filename='flask_server.log', level=logging.INFO)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# 设置超时时间
app.config['TIMEOUT'] = 60  # 设置超时时间为 60 秒

# 处理客户端断开连接
@app.after_request
def after_request(response):
    if request.endpoint == 'predict' and response.status_code == 200:
        try:
            response.direct_passthrough = False
        except (IOError, AttributeError):
            logging.error("Client disconnected before response could be sent.")
    return response
    

# 初始化突变检测相关变量
prev_rssi = {}
prev_env = {}
# 定义突变检测阈值
mutation_threshold = 15  
history_max_length = 500
mutation_detection_window = 10
window_size = 21
# 获取环境模型，默认使用model_env1
env_model_index = 1 

# 加载预训练模型
# model_path_base = '/home/lvshrd/LinearRegression/model/'
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

def curve_predict(device_name, rssi_value):
    if prev_env[device_name] == 0:
        distance = 10 ** ((-46.11375710739641 - rssi_value) / (10 * 3.7773717761249443))
    else:
        distance = 10 ** ((-41.29278313140723 - rssi_value) / (10 * 3.7674211157204023))
    return distance


# RSSI突变检测函数
def detect_rssi_mutation(rssi):
    first_half_avg = np.mean(rssi[:mutation_detection_window // 2])
    final_half_avg = np.mean(rssi[mutation_detection_window // 2:])
    if first_half_avg - final_half_avg> mutation_threshold:
        return -1   #代表隔墙遮挡
    elif first_half_avg - final_half_avg < -mutation_threshold:
        return 1    #代表空旷场景
    else:
        return 0
    
# 处理数据并进行预测的函数
def process_data_and_predict(name, rssi):
    global prev_env
    result = {}
    try:
        result['name'] = name
        # 清洗和滤波处理RSSI值
        cleaned_rssi = filter.clean_rssi_values(rssi)
        filtered_rssi = filter.weighted_blend_filter(cleaned_rssi, window_size=window_size, process_variance=1, measurement_variance=1000)
    
        # 转换为数组并进行归一化处理
        rssi_array = np.array(filtered_rssi).reshape(-1,1)
        # scaler = StandardScaler()
        # # 使用fit_transform()函数来同时计算平均值和标准差，并对数据进行标准化
        # normalized_rssi = scaler.fit_transform(rssi_array)
        normalized_rssi = (rssi_array - np.min(rssi_array)) / (np.max(rssi_array) - np.min(rssi_array))
        env_model = models.get(prev_env[name])
        
        if env_model is None:
            raise Exception("Model for environment 1 is not loaded.")
        # 使用模型预测距离值
        # rssi_tensor = torch.Tensor(normalized_rssi[-1]).unsqueeze(0).unsqueeze(2)
        # distance = env_model(rssi_tensor).item()
        # distance = distance * 4.69 + 5.5

        # rssi_tensor = torch.tensor(normalized_rssi, dtype=torch.float32)
        # distance_tensor = env_model(rssi_tensor)
        # distance_tensor = distance_tensor * 4.69 + 5.5
        # distance_numpy = distance_tensor.detach().numpy()
        # distance = distance_numpy[-1]
        rssi_tensor = torch.tensor(normalized_rssi, dtype=torch.float32)
        distance_tensor = env_model(rssi_tensor)
        distance_tensor = distance_tensor * 9 + 1
        distance = distance_tensor[-1].item()  # 获取最新的预测距离值

        distance = curve_predict(name, filtered_rssi[-1])

        if len(cleaned_rssi) >= mutation_detection_window:
            # 检查RSSI突变情况
            mutation_detected = detect_rssi_mutation(cleaned_rssi[-mutation_detection_window:])
            if mutation_detected == 1:
                # RSSI突变情况，根据情况选择不同的模型进行预测
                prev_env[name] = 1
                logging.info(f"{name}'s Model Change:{prev_env[name]}")
            elif mutation_detected == -1:
                prev_env[name] = 0
                logging.info(f"{name}'s Model Change:{prev_env[name]}")           

        result['wallStatus'] = prev_env[name]
        result['distance'] = distance
        result['rssi'] = rssi[-1]
    except Exception as e:
        result['error'] = str(e)
    logging.info(f"sent data: name={name}|rssi={rssi[-1]}|distance={distance}")
    print(f"sent data: name={name}|rssi={rssi[-1]}|distance={distance}")
    return jsonify(result)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global prev_rssi, prev_env
    # 从 Android 应用程序发送的数据
    data = request.json  # 假设数据以 JSON 格式发送

    # 获取设备名称和当前 RSSI 值
    name = data.get('name')
    mac_address = data.get('mac')
    current_rssi = data.get('rssi')

    logging.info(f"Received data: name={name}, mac={mac_address}, rssi={current_rssi}")
    # 初始化返回结果
    result = {}

    # 检查是否已经收到过该设备的数据，如果没有，则初始化其对应的 prev_rssi 列表
    if name not in prev_rssi:
        prev_rssi[name] = []
        prev_env[name] = env_model_index

    # 更新该设备对应的 prev_rssi 列表
    prev_rssi[name].append(current_rssi)
    if len(prev_rssi[name]) > history_max_length:
        prev_rssi[name] = prev_rssi[name][-history_max_length:]

    # 处理数据并进行预测
    result = process_data_and_predict(name, prev_rssi[name])
    # print(f"Sent data: {result}")
    # logging.info(f"Sent data: {result}")
    return result

if __name__ == '__main__':
    app.run(host='172.16.24.230', port=5000,debug=False)
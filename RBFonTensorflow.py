import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression(n_samples=1000, n_features=1, noise=0.1, random_state=0)

    # 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 数据集准备和预处理
# X_train = ...  # RSSI 数据
# y_train = ...  # 距离数据

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 构建 RBF 神经网络模型
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(1, activation='linear')
])

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 评估模型
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

# 应用模型
predicted_distance = model.predict(new_rssi_values)
# 在 Android 上部署模型时，可以保存模型为 TensorFlow Lite 格式
# model.save("model.h5")  # 保存为 Keras 模型
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("model.tflite", "wb").write(tflite_model)

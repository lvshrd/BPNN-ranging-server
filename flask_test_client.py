import tkinter as tk
from tkinter import ttk
import requests
import time

def send_test_data(name, mac_address, rssi):
    url = "http://172.16.24.230:5000/predict"
    data = {
        "name": name,
        "mac": mac_address,
        "rssi": rssi
    }
    response = requests.post(url, json=data)
    return response.json()

def update_rssi():
    rssi = rssi_scale.get()
    result_label.config(text=f"RSSI: {rssi}")
    result = send_test_data("TestDevice", "00:11:22:33:44:55", rssi)
    result_label.config(text=f"Server response: {result}")
    root.after(1000, update_rssi)  # 每隔0.5秒更新一次

root = tk.Tk()
root.title("RSSI Test Client")
rssi_var = tk.IntVar(value=-40)  # Set initial value here
rssi_scale = ttk.Scale(root, from_=-20, to=-80, orient="horizontal", length=200 , variable=rssi_var)
rssi_scale.pack(padx=20, pady=10)

result_label = tk.Label(root, text="RSSI: ")
result_label.pack(pady=10)

update_rssi()  # 启动更新 RSSI 数据的函数

root.mainloop()

from flask import Flask
import numpy as np
# 1. 定义app
app = Flask(__name__)
# 2. 定义函数
@app.route('/')
def hello_world():
 return 'hello,word!'
# 3. 定义ip和端口
if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8080)
    list = [0,1,2,3,4,5,6,7,8,9]
    print(np.mean(list))
    print(list[5][10])

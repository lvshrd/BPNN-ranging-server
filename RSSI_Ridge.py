import numpy as np
from sklearn.linear_model import Ridge
from data import DataProcessor


class model():
    def __init__(self, features, labels, alpha):
        self.features = features
        self.labels = labels
        self.alpha = alpha
    
    # 岭回归拟合
    def ridge_regression(self):
        # 创建岭回归模型
        ridge_model = Ridge(alpha= self.alpha)

        # 拟合岭回归模型
        ridge_model.fit(self.features, self.labels)

        # 获取回归系数
        coefficients = ridge_model.coef_

        return coefficients


if __name__ == '__main__':
    data_path = 'D:/311/FYP/my_project/Linear Regression/dataset'
    file_names = ['1D1.txt', '3D1.txt', '5D1.txt']
    data_processor = DataProcessor(data_path,file_names)
    features,labels = data_processor.read_data()
    ridge = model(features,labels,alpha=1)
    print(ridge.ridge_regression())
    



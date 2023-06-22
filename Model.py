import torch
import torch.nn as nn

input_size = 10
output_size = 67

# ニューラルネットワークモデルの定義
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # 入力層から隠れ層への線形変換
        self.relu1 = nn.ReLU()                          # ReLU活性化関数
        self.fc2 = nn.Linear(32, 64)  # 入力層から隠れ層への線形変換
        self.relu2 = nn.ReLU()                          # ReLU活性化関数
        self.fc3 = nn.Linear(64, output_size) # 隠れ層から出力層への線形変換
        self.out = nn.Softmax(dim=0) #Softmaxで活性化

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)     # 入力を隠れ層に渡す
        x = self.dropout(x) # Dropout
        x = self.relu1(x)    # 隠れ層の出力を活性化関数に渡す
        x = self.fc2(x)     # 隠れ層の出力を隠れ層に渡す
        x = self.dropout(x) # Dropout
        x = self.relu2(x)     # 隠れ層の出力を活性化関数に渡す
        x = self.fc3(x)     # 隠れ層の出力を出力層に渡す
        x = self.out(x)
        return x
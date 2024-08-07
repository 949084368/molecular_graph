# -*- coding: UTF-8 -*-
'''
@Project ：AHR_GNN 
@File    ：cattest.py
@Author  ：Mental-Flow
@Date    ：2024/4/28 10:27 
@introduction :
'''
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # 定义Transformer Encoder层
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=100, nhead=5)

    def forward(self, input_data):
        # 将输入数据转换为Transformer Encoder的输入格式
        input_data = input_data.permute(1, 0, 2)

        # 编码处理
        encoder_output = self.transformer_encoder(input_data)

        # 提取第一个token对应的输出tensor
        output = encoder_output[0]

        return output

n=20
# 输入数据
input_data = torch.randn(50, n, 100)

# 实例化Transformer Encoder模型
model = TransformerEncoder()

# 调用forward方法，获取输出
output = model(input_data)

print(output.shape)
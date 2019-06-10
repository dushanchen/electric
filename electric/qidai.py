#!/usr/bin/python 
# -*- coding: utf-8 -*-
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG

import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.nn
import torch.utils.data as Data
# import torchvision      # 数据库模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from torch.autograd import Variable


torch.manual_seed(7)

f1 = open('thunderman.csv',encoding="utf-8")
df = pd.read_csv(f1)

# df = pd.DataFrame()
# df = df1

df=df[~df['date'].isin(['2018-9-15'])]

df['week']  =  df['date']
df['week'] = pd.to_datetime(df['week'])
df['week'] = df['week'].dt.dayofweek
days = {0:'Mon',1:'Tue',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}  #哑变量尝试
df['week'] = df['week'].apply(lambda x: days[x])
#上班和不上班
df['work'] = df['week']
works = {'Mon':'Y','Tue':'Y','Wed':'Y','Thur':'Y','Fri':'Y','Sat':'N','Sun':'N'}
df['work'] = df['work'].apply(lambda x: works[x])
#节假日
df.loc[df.date=='2018-09-24','work'] = 'N'
df.loc[df.date=='2018-09-29','work'] = 'Y'
df.loc[df.date=='2018-09-30','work'] = 'Y'
df.loc[df.date=='2018-10-01','work'] = 'N'
df.loc[df.date=='2018-10-02','work'] = 'N'
df.loc[df.date=='2018-10-03','work'] = 'N'
df.loc[df.date=='2018-10-04','work'] = 'N'
df.loc[df.date=='2018-10-05','work'] = 'N'
df.loc[df.date=='2018-12-29','work'] = 'Y'
df.loc[df.date=='2018-12-31','work'] = 'N'

works1 = {'Y':1,'N':0}
df['work'] = df['work'].apply(lambda x: works1[x])
# print(df)


input_list = []

df = df.reset_index()
# print(df)
i = 0
while i < len(df['work']):
    small_list = []
    small_list.append(df['temp_max'][i])
    small_list.append(df['temp_min'][i])
    small_list.append(df['work'][i])
    input_list.append(small_list)
    i += 1


df1 = pd.DataFrame()
# for item in df['input']:
#     blank = []
#     blank.append(item[0][0])
#     blank.append(item[1][0])
#     blank.append(item[2][0])
#     item = blank
#     newlist.append(blank)
df1 = pd.DataFrame()
df1['input'] = input_list
df1['quantity'] = df['quantity']
df1['temp_max'] = df['temp_max']
df1['temp_min'] = df['temp_min']
df1['work'] = df['work']
print(df1)
df1 = df1.reset_index()

# for item in df1['input']:
#     for element in item:
#         element = [element]

# for element in df1['quantity']:
#     element = [element]


print('神经网络构建开始！')
print('开始导入数据！')
x =  torch.FloatTensor(df1['input'])
# x1 = torch.FloatTensor(df1['temp_max'])
# x2 = torch.FloatTensor(df1['temp_min'])
# x3 = torch.FloatTensor(df1['work'])
y = torch.FloatTensor(df1['quantity'])
# x = Variable(torch.Tensor(df1['input']))
# y = Variable(torch.Tensor(df1['quantity']))
print(x)
print(y)
# torch_dataset = Data.TensorDataset(x,y)
print('成功导入数据！')

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x =  F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=3, n_hidden =20 ,n_output=1)
net = net.cuda()

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()       # 预测值和真实值的误差计算公式 (均方差)

# BATCH_SIZE = 10    # 批训练的数据个数
# # 把 dataset 放入 DataLoader
# loader = Data.DataLoader(
#     dataset=torch_dataset,      # torch TensorDataset format
#     batch_size=BATCH_SIZE,      # mini batch size
#     shuffle=True,               # 要不要打乱数据 (打乱比较好)
#     num_workers=2,              # 多线程来读数据
# )
x = x.cuda()
y = y.cuda()

print('开始训练！')
start = time.time()
for epoch in range(10):
    rank = []
    
    prediction = net(x)  # 喂给 net 训练数据 x, 输出分析值
    prediction = prediction.cpu()
    print(prediction)
    prediction = prediction.cuda()
    loss = loss_func(prediction, y)     # 计算两者的误差
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    end = time.time()
    # loss = loss.cpu()
    # print(loss.item())
        # print('Epoch: ', epoch+1, '| Step: ', step+1, '| Loss: ', loss.data.numpy(),' |训练时间: ',round(end-start,2),'秒')
    # rank.append(loss.data.numpy())
    loss = loss.cuda()
    # x1 = x1.cpu()
    # x2 = x2.cpu()
    # x3 = x3.cpu()
    # x = x.cpu()
    # y = y.cpu()
    loss = loss.cpu()
            # # 过了一道 softmax 的激励函数后的最大概率才是预测值
            # prediction = torch.max(F.softmax(out), dim=1)[1]
            # prediction1 = prediction.cpu()
            # pred_y = prediction1.data.numpy().squeeze()
            # target_y = batch_y.data.numpy()
    # average = float(sum(rank)/len(rank))
    print('Epoch: ', epoch+1, '| Loss: ', loss,' |训练时间: ',round(end-start,2),'秒')
    # print('---------------------------------------------')
    # print('保存第',epoch+1,'遍训练模型中...')
    # print('---------------------------------------------')
    # torch.save(net.state_dict(), 'regnet.pt') # 保存整个网络参数

print('神经网络结束！')
end = time.time()
print('总用时: ',end-start,'秒')


torch.manual_seed(7)
torch.save(net.state_dict(), 'regnet.pt') # 保存整个网络参数
print('保存成功！')



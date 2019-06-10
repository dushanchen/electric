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


# torch.manual_seed(7)

f1 = open('thunderman.csv',encoding="utf-8")
df = pd.read_csv(f1)
df = df.reset_index()
df=df[~df['date'].isin(['2018-9-15'])]
df = df.reset_index()
# print(df)

i = 0
input_list = []
while i < len(df['work']):
    small_list = []
    small_list.append(df['temp_max'][i])
    small_list.append(df['temp_min'][i])
    input_list.append(small_list)
    i += 1


df1 = pd.DataFrame()
df1['input'] = input_list
df1['quantity'] = df['quantity']
df1['temp_max'] = df['temp_max']
df1['temp_min'] = df['temp_min']
df1['work'] = df['work']
print(df1)
df1 = df1.reset_index()

df2 = pd.DataFrame()
# work = 0
df2 = df1[df1['work'].isin([0])]
df2 = df2.reset_index()


df3 = pd.DataFrame()
# work = 1
df3['input']  =  input_list
df3 = df1[df1['work'].isin([1])]
df3 = df3.reset_index()


print('神经网络构建开始！')
print('开始导入数据！')
# x =  torch.FloatTensor(df3['temp_max'])
# y = torch.FloatTensor(df3['quantity'])
# x = torch.unsqueeze(x, dim=1)
# y = torch.unsqueeze(y, dim=1)
# x, y = Variable(x), Variable(y)

x1_mean = np.mean(df3['temp_max'])
x1_div = np.std(df3['temp_max'],ddof=1)
x2_mean = np.mean(df3['temp_min'])
x2_div = np.std(df3['temp_min'],ddof=1)
y_mean = np.mean(df3['quantity'])
y_div = np.std(df3['quantity'],ddof=1)

df0 = pd.DataFrame()
df0['x1_mean'] = [x1_mean]
df0['x1_div'] = [x2_div]
df0['x2_mean'] = [x2_mean]
df0['x2_div'] = [x2_div]
df0['y_mean'] = [y_mean]
df0['y_div'] = [y_div]
df0 = df0.reset_index()

df0.to_csv('stat1.csv', encoding='utf-8', index = False)



for i in df3['input']:
    i[0] = (i[0]-x1_mean)/x1_div
    i[1] = (i[1]-x2_mean)/x2_div
x = df3['input']

for i in df3['quantity']:
    i = (i - y_mean)/y_div
y = df3['quantity']

y = torch.FloatTensor([(_-y_mean)/y_div for _ in df3['quantity']])

x =  torch.FloatTensor([x]).squeeze(2)
# y = torch.FloatTensor([y])

x = torch.squeeze(x, dim=1)
y = torch.unsqueeze(y, dim=1)
x, y = Variable(x), Variable(y)



# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.FloatTensor(np_data)
# x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
# y = x.pow(3) + 0.2*torch.rand(x.size())
# x, y = Variable(x), Variable(y)



# print(x.dtype)
print(x)
# print(y.dtype)
print(y)
print('成功导入数据！')

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x =  F.relu6(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=2, n_hidden =18 ,n_output=1)

# plt.ion()
# plt.show()
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()       # 预测值和真实值的误差计算公式 (均方差)

# y =  y.view(-1,1,1)

print('开始训练！')
start = time.time()
for epoch in range(100000):
    # rank = []
    
    prediction = net(x)  # 喂给 net 训练数据 x, 输出分析值
    # print(prediction)
    loss = loss_func(prediction, y)     # 计算两者的误差
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    end = time.time()
    # loss = loss.cpu()
    # print(loss.item())
        # print('Epoch: ', epoch+1, '| Step: ', step+1, '| Loss: ', loss.data.numpy(),' |训练时间: ',round(end-start,2),'秒')
    # rank.append(loss.data.numpy())
    # x1 = x1.cpu()
    # x2 = x2.cpu()
    # x3 = x3.cpu()
    # x = x.cpu()
    # y = y.cpu()

    # average = float(sum(rank)/len(rank))

    # if epoch % 10 ==0:
    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size':'20','color':'red'})
    #     plt.pause(0.1)


    if epoch % 100 ==0:
        # print(prediction)
        print('Epoch: ', epoch+1, '| Loss: ', loss.item(),' |训练时间: ',round(end-start,2),'秒')
    # print(prediction)
    # print('---------------------------------------------')
    # print('保存第',epoch+1,'遍训练模型中...')
    # print('---------------------------------------------')
    # torch.save(net.state_dict(), 'regnet.pt') # 保存整个网络参数

plt.ioff()
plt.show()


print('神经网络结束！')
end = time.time()
print('总用时: ',end-start,'秒')


# torch.manual_seed(7)
torch.save(net.state_dict(), 'regnet1.pt') # 保存整个网络参数
# print('保存成功！')



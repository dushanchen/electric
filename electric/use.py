#!/usr/bin/python 
# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这
import pandas as pd
import datetime
import time 
import jieba
import numpy as np
import tkinter as tk
import random



def norm_raw(raw_list):
    if raw_list[2] == 1:
        # workday
        f1 = open('stat1.csv',encoding="utf-8")
        df = pd.read_csv(f1)
        df = df.reset_index()
        list1 = []
        list1.append((raw_list[0]-df['x1_mean'][0])/(df['x1_div'][0]))
        list1.append((raw_list[1]-df['x2_mean'][0])/(df['x2_div'][0]))
        return list1
    elif raw_list[2] == 0:
        # freeday
        f1 = open('stat2.csv',encoding="utf-8")
        df = pd.read_csv(f1)
        df = df.reset_index()
        list1 = []
        list1.append((raw_list[0]-df['x1_mean'][0])/(df['x1_div'][0]))
        list1.append((raw_list[1]-df['x2_mean'][0])/(df['x2_div'][0]))
        return list1

def norm_result(result, raw_list):

    if raw_list[2] == 1:
        # workday
        f1 = open('stat1.csv',encoding="utf-8")
        df = pd.read_csv(f1)
        df = df.reset_index()
        result1 = result * df['y_div'][0]+df['y_mean'][0]
        return result1
    elif raw_list[2] == 0:
        # freeday
        f1 = open('stat2.csv',encoding="utf-8")
        df = pd.read_csv(f1)
        df = df.reset_index()
        result1 = result * df['y_div'][0]+df['y_mean'][0]
        return result1

def pred(con):
    if con[2] == 1:
        # workday
        con = norm_raw(con)
        x = torch.FloatTensor(con) 
        class Net(torch.nn.Module):
            def __init__(self, n_feature, n_hidden, n_output):
                super(Net, self).__init__()
                self.hidden = torch.nn.Linear(n_feature, n_hidden)
                self.predict = torch.nn.Linear(n_hidden, n_output)
            
            def forward(self, x):
                x =  F.relu(self.hidden(x))
                x = self.predict(x)
                return x
        # optimizer 是训练的工具
        net = Net(n_feature=2, n_hidden =18 ,n_output=1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
        loss_func = torch.nn.MSELoss()       # 预测值和真实值的误差计算公式 (均方差)
        net.load_state_dict(torch.load('regnet1.pt', map_location = 'cpu'))
        print('模型导入成功！')
        prediction = net(x)
        con.append(1)
        prediction = norm_result(prediction.data.numpy()[0],con)
        return prediction

    elif con[2] == 0:
        # freeday
        con = norm_raw(con)
        x = torch.FloatTensor(con) 
        class Net(torch.nn.Module):
            def __init__(self, n_feature, n_hidden, n_output):
                super(Net, self).__init__()
                self.hidden = torch.nn.Linear(n_feature, n_hidden)
                self.predict = torch.nn.Linear(n_hidden, n_output)
            
            def forward(self, x):
                x =  F.relu(self.hidden(x))
                x = self.predict(x)
                return x
        # optimizer 是训练的工具
        net = Net(n_feature=2, n_hidden =18 ,n_output=1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
        loss_func = torch.nn.MSELoss()       # 预测值和真实值的误差计算公式 (均方差)
        net.load_state_dict(torch.load('regnet2.pt', map_location = 'cpu'))
        # print('模型导入成功！')
        prediction = net(x)
        con.append(0)
        prediction = norm_result(prediction.data.numpy()[0], con)
        return prediction


a = [29, 23, 0]

pred(a)
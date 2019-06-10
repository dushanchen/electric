from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .use import *
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
import time
import re

# Create your views here.

@csrf_exempt
def index(request):

    ctx = {}
    

    if request.method  == "POST":
        ctx['max'] = max =  request.POST.get('max', '')
        ctx['min'] = min =  request.POST.get('min', '')
        ctx['work'] = work =  request.POST.get('work', '')

        print(max)
        print(min)
        print(work)
        # 开始计算

        
        list1 = [int(max), int(min), int(work)]
        result = str(pred(list1))
        print(result)
        ctx['result']  = round(float(result),1)
        return JsonResponse(ctx)
  
    a, b, c, d = aa()
    chart =  []
    for i in range(7):
        max_ = a[i]
        min_ = b[i]
        work = c[i]
        print(max_, min_, work)
        list1 = [max_, min_, work]
        r = str(pred(list1))
        chart.append({
            'max':max_,
            'min':min_,
            'work': '休假' if work == 0 else '工作',
            'day':d[i],
            'count':round(float(r),1)
        })
    ctx['chart'] = chart


    return render(request, 'index.html', ctx)



def aa():
    import json
    import requests
    import datetime
    from pyquery import PyQuery as p


    url = 'http://www.weather.com.cn/weather/101021200.shtml'
    result = requests.get(url)

    a = p(result.text).find('.sky .tem')

    max_list = []
    min_list = []
    work_list = []
    date_list = []

    for i in a:
        # print(p(i).html())
        max = p(i).find('span').html()
        min = p(i).find('i').html()
        if not max:
            max = '32'
        max =  re.findall(re.compile('\d+'), max)[0]
        min =  re.findall(re.compile('\d+'), min)[0]
        print(max, min)
        max_list.append(int(max))
        min_list.append(int(min))

    today = datetime.date.today()
    holiday_url = 'http://api.goseek.cn/Tools/holiday?date='
    #https://www.jianshu.com/p/05ccb5783f65
    work = 0

    for i in range(7):
        s = today.strftime('%Y%m%d')
        date_list.append(s)

        url = holiday_url + s
        res = requests.get(url).text
        print(res)
        res = json.loads(res)


        if 'data' in res:
            if res['data'] == 2 or res['data'] == 0:
                work = 1 # 上班
            else:
                work = 0 # 不上班

        today = today + datetime.timedelta(days=1)
        print(s, work)
        work_list.append(work)
        # time.sleep(0.5)

    return max_list, min_list, work_list, date_list
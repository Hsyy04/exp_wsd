import torch.nn as nn
import json
import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
import xml.dom.minidom
from aip import AipNlp
from train import Net
import json
import jieba

# 为了使用百度的转化词向量工具
APP_ID = "Hsyy04"
API_KEY = "57f8721108384b0aac655472f2d9b921"
SECRET_KEY = '6d213be9b1d044d7a03505770cbd2b30'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

def std_key(model):
    model.cpu()
    model.eval()
    dataX = []
    with open("datavec/jiao_test.json","r") as f:
        dataX=json.load(f)

    fkey = open("testK.key","w+")
    idx = 0
    for data in dataX:
        idx+=1
        input = torch.tensor(data)
        input = input.reshape(1,10,40)
        output = model(input)
        print(output)
        tar = output.max(1)[1].item()
        map_id = ['ask','call','name','cry']
        tar = map_id[int(tar)]
        fkey.write("叫 叫."+"%02d "%(idx)+str(tar)+'\n')
    fkey.close()
        

class task_demo():
    def __init__(self, char_str, model):
        self.model = model
        self.data = []
        words = jieba.lcut(char_str)
        for wd in words:
            try:
                vect = client.wordEmbedding(wd)['vec']
                self.data.append(vect)
            except:
                continue
        pass

    def wsd(self,words):
        tar_vec = self.model(self.data)
        tar = torch.max(tar_vec)
        map_id = ['ask','call','name','cry']
        tar = map_id[tar]
        return tar

model = torch.load("net/jiao.pkl")
std_key(model)
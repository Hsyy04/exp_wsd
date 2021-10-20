import torch.nn as nn
import json
import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from keras.utils.np_utils import to_categorical
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
# from tensorboardX import SummaryWriter

bactch_size=16
epoch_num = 20

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.LSTM=nn.LSTM(input_size=40,hidden_size=32,bidirectional=True,batch_first=True)
        self.fc=nn.Linear(64*10,4)

    def forward(self,input):
        x,_=self.LSTM(input)
        x=x.reshape(len(x),-1)
        x= self.fc(x)
        x=F.log_softmax(x,dim=1)
        return x

def train(loader, model, optimizer, epoch):
    model.train()
    idx=0
    for data,target in loader:
        idx+=1

        # 使用CPU需要改这里
        data,target=data.cuda(),target.cuda()

        # 计算loss
        data, target = Variable(data,requires_grad=True), Variable(target)
        output = model(data)
        loss = F.nll_loss(output, target,size_average=False) 

        #改变参数
        loss.backward()
        optimizer.step()

        if idx % 3==0:            # 输出信息
            print('Train Epoch{}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * bactch_size, len(loader.dataset),
                100. * idx / len(loader), loss.item()/len(loader.dataset)))

def proc_data():
    dataX = []
    dataY = []
    data = []
    class_flag = {'ask':0,'call':1,'name':2,'cry':3}

    # 读入数据
    with open("datavec/jiao_data.json","r") as f:
        dataX=json.load(f)

    with open("datavec/jiao_ans.json","r") as f:
        data=json.load(f)
        for i in data :
            dataY.append(class_flag[i])
    # dataY=to_categorical(dataY)

    # 创建数据集
    # print(type(dataX))
    # print(torch.tensor(dataX))
    dataset = TensorDataset(torch.tensor(dataX), torch.tensor(dataY))
    my_data = DataLoader(dataset=dataset, batch_size=bactch_size, shuffle=True)

    return my_data

def train_test(loader, model):
    model.eval()
    loss = 0
    ok = 0
    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss += F.nll_loss(output, target,size_average=False).item()
        pred = output.max(1)[1]-target
        for i in pred:
            if i==0:
                ok+=1
        
    loss/=len(loader.dataset)
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss, ok, len(loader.dataset), 100. * ok / len(loader.dataset)))

if __name__ == "__main__":
    model = Net()
    data = proc_data()
    model.cuda()
    opt = SGD(model.parameters(), lr=0.01)

    for i in range(epoch_num):
        train(data, model, opt, i)
        train_test(data, model)

    torch.save(model, "net/jiao.pkl") 
    pass
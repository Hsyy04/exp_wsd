from keras.utils.np_utils import to_categorical
import json
dataX = []
dataY = []
data = []
class_flag = {'ask':0,'call':1,'name':2,'cry':3}


with open("datavec/jiao_data.json","r") as f:
    dataX=json.load(f)

with open("datavec/jiao_ans.json","r") as f:
    data=json.load(f)
    for i in data:
        print(type(i))
        dataY.append(class_flag[i])

dataY=to_categorical(dataY)
print(dataY)
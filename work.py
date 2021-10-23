import json
import torch
from train import Net
import json
import jieba.posseg as pseg
import jieba
import argparse
jieba.setLogLevel(jieba.logging.INFO)

def std_key(model):
    # 这是一个用于生成测试答案的函数
    model.cpu()
    model.eval()
    dataX = []
    with open("datavec/jiao_test.json","r") as f:
        dataX=json.load(f)

    fkey = open("answer/testK.key","w+")
    idx = 0
    for data in dataX:
        idx+=1
        input = torch.tensor(data)
        input = input.reshape(1,6,40)
        output = model(input)
        tar = output.max(1)[1].item()
        map_id = ['ask','name','call','cry']
        # with open()
        tar = map_id[int(tar)]
        fkey.write("叫 叫."+"%02d "%(idx)+str(tar)+'\n')
    fkey.close()
        

class task_demo():
    def __init__(self, word, model):
        self.model = model
        self.model.cpu()
        self.word = word
        pass

    def wsd(self, char_str):
        # POS
        pos_dict = {'Ag':0,'a':1,'ad':2,'an':3,'b':4,'c':5,'Dg':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'Ng':17,'n':18,'nr':19,'ns':20,'nt':21,'nz':22,'o':23,'p':24,'q':25,'r':26,'s':27,'Tg':28,'t':29,'u':30,'Vg':31,'v':32,'vd':33,'vn':34,'w':35,'x':36,'y':37,'z':38,'nx':39}

        lfenci = pseg.lcut(char_str)
        tarid = lfenci.index(pseg.pair('叫', 'v'))
        self.data = []
        # 前文
        if tarid-3 < 0:
            for i in range(3-tarid):
                self.data.append([0.0 for i in range(40 )])
        for i in range(max(tarid-3,0),tarid):
            vect=[0.0 for i in range(40 )]
            try:
                x = pos_dict[lfenci[i].flag]
            except:
                lfenci[i].flag = lfenci[i].flag[0]
            vect[pos_dict[lfenci[i].flag]]=1.0
            self.data.append(vect)

        # 后文(包括了本身)
        for i in range(tarid,min(tarid+3,len(lfenci))):
            vect=[0.0 for i in range(40 )]
            try:
                x = pos_dict[lfenci[i].flag]
            except:
                lfenci[i].flag = lfenci[i].flag[0]
            vect[pos_dict[lfenci[i].flag]]=1.0
            self.data.append(vect)
        if tarid+3 > len(lfenci):
            for i in range(tarid+3-len(lfenci)):
                self.data.append([0.0 for i in range(40 )])
        self.data = torch.tensor(self.data).reshape(1, 6, 40)
        output = self.model(self.data)
        tar = output.max(1)[1].item()

        with open("datavec/"+self.word+"_flag.json","r") as f:
            class_flag = json.load(f)
            map_id = list(class_flag.keys())
        print(tar)
        tar = map_id[int(tar)]
        return tar


parser = argparse.ArgumentParser(description='中文词义消歧系统')
parser.add_argument('--wd', default='叫', help='需要消歧的词语')
parser.add_argument('--st', default='就是这卤牛肉，它能叫人白手起家成万元、十万元、百万元户', help='需要消歧的句子')
parser.add_argument('--test', default=False, help='是否生成测试答案')
args = parser.parse_args()

model = torch.load("net/"+args.wd+".pkl")
if args.test:
    std_key(model)

demo = task_demo(args.wd,model)
print(demo.wsd(args.st))


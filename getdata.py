import xml.dom.minidom
from aip import AipNlp
import json

# 为了使用百度的转化词向量工具
APP_ID = "Hsyy04"
API_KEY = "57f8721108384b0aac655472f2d9b921"
SECRET_KEY = '6d213be9b1d044d7a03505770cbd2b30'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
# POS
pos_dict = {'Ag':0,'a':1,'ad':2,'an':3,'b':4,'c':5,'Dg':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'Ng':17,'n':18,'nr':19,'ns':20,'nt':21,'nz':22,'o':23,'p':24,'q':25,'r':26,'s':27,'Tg':28,'t':29,'u':30,'Vg':31,'v':32,'vd':33,'vn':34,'w':35,'x':36,'y':37,'z':38,'nx':39}

# 打开训练数据
dom = xml.dom.minidom.parse('trial\corpus\Chinese_test.xml')
root = dom.documentElement

# 获取所有歧义词语
lexelt = root.getElementsByTagName('lexelt')

data ={}
cnt = 0

# 获取每一个歧义词语的例句
for word in lexelt:
    instance = word.getElementsByTagName("instance")
    item = word.getAttribute("item")
    dataX = [] #
    dataY = [] #

    for ins in instance: 
        # 这个例子里歧义词语的含义
        ans = ins.getElementsByTagName("answer")
        sense = ans[0].getAttribute("senseid")
        dataY.append(sense)
        
        # 标注信息 
        postagging = ins.getElementsByTagName("postagging")[0]

        # 获取上下文例句

        # 我想直接获得这个例句词性的向量化值, 每句话应该对应40*10的矩阵
        # 目标词语的前5个和后5个词
        pos = []
        tokens = []
        for i in postagging.getElementsByTagName("word"):
            pos.append(i.getAttribute("pos"))
            tokens.append(i.getElementsByTagName("token")[0].firstChild.data)
        tarid = tokens.index('叫')

        X = []
        # 前文
        if tarid-5 < 0:
            for i in range(5-tarid):
                X.append([0.0 for i in range(40 )])
        for i in range(max(tarid-5,0),tarid):
            vect=[0.0 for i in range(40 )]
            vect[pos_dict[pos[i]]]=1.0
            X.append(vect)

        # 后文(包括了本身)
        for i in range(tarid,min(tarid+5,len(pos))):
            vect=[0.0 for i in range(40 )]
            vect[pos_dict[pos[i]]]=1.0
            X.append(vect)
        if tarid+5 > len(pos):
            for i in range(tarid+5-len(pos)):
                X.append([0.0 for i in range(40 )])

        dataX.append(X)
    break

print("data win")

with open("datavec/jiao_test.json","w+") as f:
    json.dump(dataX, f)


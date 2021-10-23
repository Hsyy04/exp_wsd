import xml.dom.minidom
import json
import os
# POS
pos_dict = {'Ag':0,'a':1,'ad':2,'an':3,'b':4,'c':5,'Dg':6,'d':7,'e':8,'f':9,'g':10,'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,'Ng':17,'n':18,'nr':19,'ns':20,'nt':21,'nz':22,'o':23,'p':24,'q':25,'r':26,'s':27,'Tg':28,'t':29,'u':30,'Vg':31,'v':32,'vd':33,'vn':34,'w':35,'x':36,'y':37,'z':38,'nx':39}
word_pool = []

# 打开训练数据
# dom = xml.dom.minidom.parse('trial\corpus\Chinese_train.xml')
dom = xml.dom.minidom.parse('train\Chinese_train_pos.xml')
root = dom.documentElement

# 获取所有歧义词语
lexelt = root.getElementsByTagName('lexelt')

data ={}
cnt = 0
bk = 0

# 获取每一个歧义词语的例句
for word in lexelt:
    instance = word.getElementsByTagName("instance")
    item = word.getAttribute("item")
    word_pool.append(item)
    dataX = [] # 该词语对应的输入向量
    dataY = [] #该词语对应的输出类别
    flag_cnt = 0
    word_flag = {}
    for ins in instance: 
        # 这个例子里歧义词语的含义
        ans = ins.getElementsByTagName("answer")
        sense = ans[0].getAttribute("senseid")
        dataY.append(sense)
        try  :
            word_flag[sense]
        except:
            word_flag[sense] = flag_cnt
            flag_cnt+=1
        # 标注信息 
        postagging = ins.getElementsByTagName("postagging")[0]

        # 直接获得这个例句词性的向量化值, 每句话应该对应40*10的矩阵
        # 目标词语的前5个和后5个词
        pos = []
        tokens = []
        for i in postagging.getElementsByTagName("word"):
            if len(i.getElementsByTagName("subword"))!=0:
                for j in i.getElementsByTagName("subword"):
                    pos.append(j.getAttribute("pos"))
                    tokens.append(j.getElementsByTagName("token")[0].firstChild.data)
            else:
                pos.append(i.getAttribute("pos"))
                tokens.append(i.getElementsByTagName("token")[0].firstChild.data)
        # print(ans[0].getAttribute("instance"))
        tarid = tokens.index(item)

        X = []
        # 前文
        if tarid-5 < 0:
            for i in range(3-tarid):
                X.append([0.0 for i in range(40 )])
        for i in range(max(tarid-3,0),tarid):
            vect=[0.0 for i in range(40 )]
            tem = ''
            try:
                tem = pos_dict[pos[i]]
            except:
                tem = pos_dict[pos[i][0]]
            vect[tem]=1.0
            X.append(vect)

        # 后文(包括了本身)
        for i in range(tarid,min(tarid+3,len(pos))):
            vect=[0.0 for i in range(40 )]
            tem = ''
            try:
                tem = pos_dict[pos[i]]
            except:
                tem = pos_dict[pos[i][0]]
            vect[tem]=1.0
            X.append(vect)
        if tarid+3 > len(pos):
            for i in range(tarid+3-len(pos)):
                X.append([0.0 for i in range(40 )])

        dataX.append(X)
    
    # 保存向量化后的数据
    path_data = os.path.join("datavec",str(item)+"_data.json")
    path_ans = os.path.join("datavec",str(item)+"_ans.json")
    path_flag = os.path.join("datavec",str(item)+"_flag.json")
    with open(path_data,"w+") as f:
        json.dump(dataX, f)

    with open(path_ans,"w+") as f:
        json.dump(dataY, f)
    
    with open(path_flag,"w+") as f:
        json.dump(word_flag, f)

# 保存数据里的所有词汇
with open("datavec/word_pool.json","w+") as f:
    json.dump(word_pool,f)

print("data win！")


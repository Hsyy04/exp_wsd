# 简单中文词义消歧系统

## 支持消歧词语
叫: {"ask": 0, "name": 1, "call": 2, "cry": 3}
处：{"aspect": 0, "place": 1, "department": 2} 

## Demo 运行方式

    py work.py --wd [需要消歧的词语] --st [需要消歧的句子]
    or 
    python3 work.py --wd [需要消歧的词语] --st [需要消歧的句子]

例如，
```bash
    py work.py --wd 叫 --st  就是这卤牛肉，它能叫人白手起家成万元、十万元、百万元户
    or
    python3 work.py --wd 叫 --st  就是这卤牛肉，它能叫人白手起家成万元、十万元、百万元户
```

## 数据集
SemEval-2007

## 其他注释
`work.py`      运行系统的接口
`data.py`      训练数据预处理
`train.py`     模型定义及其训练
`getdata.py`    预处理测试数据，如果运行时添加 --test 1，则会在answer文件夹下生成一个key文件。    
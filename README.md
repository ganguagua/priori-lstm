# priori-lstm
一款可以配置词典的模型，极大便利了工业应用；识别新词只需更新词典，而不用重新训练模型。
基于可配置词典的lstm切词。 将最大前向匹配的结果作为模型输入的一部分，达到动态干预模型结果的效果，实时加入新词

# 运行
1. 环境tensorflow2.0 python3.6；除此之外还有一些python包：numpy, collections
1. 首先需要下载bert预训练模型，解压到当前目录
2. controller.py是主脚本，参数维train或者predict
3. 可参考run.sh编写启动脚本

# 配置项说明
1. 配置文件：lstm.conf
2. + hidden_size: lstm层的维度 
   + max_seq_len: 最大句子长度
   + learning_rate: 学习率
   + bert_vocab: bert的词(字)库
   + pre_train_model: bert预训练的checkpoint
   + vocab_size: 字库的大小
   + train_file: 训练数据 文件路径
   + dev_file: 验证集文件路径
   + test_file: 测试集文件路径

# 数据
用的是微软亚洲研究院的公开数据集，读者可自行下载更多的数据集

# 模型效果
可以达到准确率99%，新词配置后能很好地识别

# priori-lstm
一款可以配置词典的模型，极大便利了工业应用；识别新词只需更新词典，而不用重新训练模型。
基于可配置词典的lstm切词。 将最大前向匹配的结果作为模型输入的一部分，达到动态干预模型结果的效果，实时加入新词

# 运行
1. 环境tensorflow2.0 python3.6；除此之外还有一些python包：numpy, collections
2. 直接运行run.sh即可开始训练
3. 超参在lstm.conf中修改，通过tensorboard --logdir ./tf.log/  --bind_all查看训练过程
4. 如果bert预训练模型下载缓慢，可以自行从其他地方下载，bert模型下载好后，运行controller.py即可
```
python controller.py train #训练
python controller.py predict #测试
```

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


样例：
```
城市  大  了  ，  我们  的  活  也  就  多  了  重  了  ，  可  这点儿  辛苦  真  算  不得  什么  。  ”
从小  苦  惯  了  ，  闲  下来  就  会  浑身  不  舒服  。  ”
凭  自己  的  技术  ，  总能  找到  活  干  ，  而且  能  让  人  满意  。  ”
她  跟  等候  着  的  求职  者  一一  打招呼  ：  “  你们  去  找  旁边  的  老师  登记  好  吗  ？
就业  的  天地  很  广  ，  不要  在  一  棵  树  上  吊死  ，  至少  先  解决  温饱  再说  。  ”
那  你  等  一  等  ，  我  再  帮  你  调调  资料  看  。  ”
这是  花  ，  是  树桩  ，  哪  得  是  干柴  嘛  ！  ”
```
# 模型效果
新词配置后能很好地识别

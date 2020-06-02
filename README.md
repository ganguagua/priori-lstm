# priori-lstm
基于可配置词典的lstm切词。 将最大前向匹配的结果作为模型输入的一部分，达到动态干预模型结果的效果，实时加入新词

# 运行方式
1. 首先需要下载bert预训练模型，解压到当前目录
2. controller.py是主脚本，参数维train或者predict
3. 可参考run.sh

# 数据
用的是微软亚洲研究院的公开数据集，读者可自行下载更多的数据集

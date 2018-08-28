######
音乐分离任务：chimera

参考文献：DEEP CLUSTERING AND CONVENTIONAL NETWORKS FOR MUSIC SEPARATION: STRONGER TOGETHER
(https://arxiv.org/abs/1611.06265)

######
服务器：111.230.219.182:36000

代码地址：/media/data/rainiejjli/pytorch_chimera_music_separation

数据集地址：/media/data/rainiejjli/music_dataset

trained model地址：/media/data/rainiejjli/models/chimera_model.pkl

######
脚本说明：
ProcessDSD.py/ProcessIKALA.py/ProcessMedleyDB.py: 处理数据集
config.py: 参数设置
utils.py: 提取特征
dataset.py: pytorch dataloader一个batch的数据
chimera.py: chimera模型
trainer.py: 训练模型及在验证集上验证，保存模型
train_chimera.py: 主函数，训练
seperate.py: 测试分离

######
运行过程说明：
1、提取特征
python ProcessDSD.py/ProcessIKALA.py/ProcessMedleyDB.py
2、训练模型
python train_chimera.py
3、分离测试
测试wav文件放到test文件夹中，分离后的人声与伴奏存储在enhanced文件夹中
python seperate.py
######
音乐分离任务：U_Net

参考文献：Singing Voice Separation with Deep U-Net Convolutional Networks
(http://openaccess.city.ac.uk/19289/1/7bb8d1600fba70dd79408775cd0c37a4ff62.pdf)

######
服务器：111.230.219.182:36000

代码地址：/media/data/rainiejjli/pytorch_unet_music_separation

数据集地址：/media/data/rainiejjli/music_dataset

trained model地址：/media/data/rainiejjli/unet_model.pkl

######
脚本说明：
ProcessDSD.py/ProcessIKALA.py/ProcessMedleyDB.py: 处理数据集
const.py: 参数设置
utils.py: 提取特征并存储在features/
dataset.py: pytorch dataloader一个batch的数据
unet.py: U_Net模型
trainer.py: 训练模型及在验证集上验证，保存模型
train_unet.py: 主函数，训练
seperate.py: 测试分离

######
运行过程说明：
1、提取特征
python ProcessDSD.py/ProcessIKALA.py/ProcessMedleyDB.py
2、训练模型
python train_unet.py
3、分离测试
测试wav文件放到test文件夹中，分离后的人声与伴奏存储在enhanced文件夹中
python seperate.py
######
���ַ�������chimera

�ο����ף�DEEP CLUSTERING AND CONVENTIONAL NETWORKS FOR MUSIC SEPARATION: STRONGER TOGETHER
(https://arxiv.org/abs/1611.06265)

######
��������111.230.219.182:36000

�����ַ��/media/data/rainiejjli/pytorch_chimera_music_separation

���ݼ���ַ��/media/data/rainiejjli/music_dataset

trained model��ַ��/media/data/rainiejjli/models/chimera_model.pkl

######
�ű�˵����
ProcessDSD.py/ProcessIKALA.py/ProcessMedleyDB.py: �������ݼ�
config.py: ��������
utils.py: ��ȡ����
dataset.py: pytorch dataloaderһ��batch������
chimera.py: chimeraģ��
trainer.py: ѵ��ģ�ͼ�����֤������֤������ģ��
train_chimera.py: ��������ѵ��
seperate.py: ���Է���

######
���й���˵����
1����ȡ����
python ProcessDSD.py/ProcessIKALA.py/ProcessMedleyDB.py
2��ѵ��ģ��
python train_chimera.py
3���������
����wav�ļ��ŵ�test�ļ����У����������������洢��enhanced�ļ�����
python seperate.py
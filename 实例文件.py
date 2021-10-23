import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.get_cachedir() # 清除一下matplotlib的cache
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=14) # 设置matplotlib画图支持中文
sns.set(font=myfont.get_name()) # 设置sns画图支持中文
matplotlib.rcParams['axes.unicode_minus'] = False # 解决matplotlib画图符号变方块的现象
import sys,warnings
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#试图使用GPU加速，但是最终失败了
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 导入数据集
data = pd.read_csv(r"./housing_question/creditcard.csv")
train_data = pd.read_csv(r"./housing_question/housing_train.csv")
test_data = pd.read_csv(r"./housing_question/housing_test.csv")
#data.head()
#plt.figure(figsize=(8,8))
#data['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='是否诈骗')
#plt.show()

#数据清洗部分
#因为原来的数据已经经过了PCA处理，所以我在这里认为数据不需要进行关联性之类的处理
#其中交易时间我不认为与交易欺诈存在关联
#print(data['Class'].value_counts())
#可知非诈骗284315，诈骗492，相差577倍，首先尝试使用SMOTE方法，将诈骗样本提升至这个个数（我觉得不太行的说）

#SMOTE类
class Smote(object):
    def __init__(self, N=50, k=5, r=2):
        self.N = N
        self.k = k
        self.r = r
        # self.newindex用于记录SMOTE算法已合成的样本个数
        self.newindex = 0

    def fit(self, samples):
        self.samples = samples
        # self.T是少数类样本个数，self.numattrs是样本的特征个数
        self.T, self.numattrs = self.samples.shape
        # 查看N%是否小于100%
        if (self.N < 100):
            # 如果是，随机抽取N*T/100个样本，作为新的少数类样本
            np.random.shuffle(self.samples)
            self.T = int(self.N * self.T / 100)
            self.samples = self.samples[0:self.T, :]
            # N%变成100%
            self.N = 100
        # 查看从T是否不大于近邻数k
        if (self.T <= self.k):
            self.k = self.T - 1
        # 令N是100的倍数
        N = int(self.N / 100)
        # 创建保存合成样本的数组
        self.synthetic = np.zeros((self.T * N, self.numattrs))
        # 调用并设置k近邻函数
        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(self.samples)
        # 对所有输入样本做循环
        for i in range(len(self.samples)):
            # 调用kneighbors方法搜索k近邻
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)),
                                           return_distance=False)[0][1:]
            self.__populate(N, i, nnarray)
        return self.synthetic

    def __populate(self, N, i, nnarray):
        # 按照倍数N做循环
        for j in range(N):
            # attrs用于保存合成样本的特征
            attrs = []
            nn = random.randint(0, self.k - 1)
            # 计算差值
            diff = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.uniform(0, 1)
            self.synthetic[self.newindex] = self.samples[i] + gap * diff
            self.newindex += 1

#建立我的dataset类型
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

#分类训练模型——逻辑回归算法
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(30, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 2)

        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x

#试图转入GPU
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
#绘制图形的方法
def plot_curve(record,title=''):
    steps=len(record)
    x=range(steps)
    plt.figure(figsize=(6,4))
    plt.plot(x,record)
    plt.xlabel('Steps')
    plt.ylabel('{}'.format(title))
    plt.title('curve of {}'.format(title))
    plt.show()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def AUC(test):
    probility = test[..., 0:1]
    actual = test[..., 1:2]
    tpr = []
    fpr = []
    for i in range(1, len(probility), 50):
        threshold = probility[i]
        tp, fp, fn, tn = 0, 0, 0, 0
        for j in range(len(probility)):
            if probility[j] >= threshold and actual[j] == 0:
                        tp += 1
            elif probility[j] >= threshold and actual[j] == 1:
                        fp += 1
            elif probility[j] < threshold and actual[j] == 0:
                        fn += 1
            elif probility[j] < threshold and actual[j] == 1:
                        tn += 1

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (tn + fp))
    return tpr, fpr
#实例化smote类
smote=Smote(N=50000)#500扩大
#欺诈类提取
data_chat= train_data[train_data['Class']==1]
#print(data_chat)  #然后看一下
data_more=smote.fit(data_chat.values) #data_more 是array类型

BATCH_SIZE = 64

#数据封装
train = np.vstack((train_data.values,data_more))
train_x=train[...,:-1]
train_y=train[...,30:]
train_y=train_y.reshape(436345,)

test_x = test_data.values[...,:-1]
test_y = test_data.values[...,30:]

train_set = MyDataSet(train_x, train_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = MyDataSet(test_x,None)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)

#GPU查看
same_seeds(0)
device = get_device()
print(f'DEVICE: {device}')
#人工数据设定
num_epoch = 15
learning_rate = 0.00001
#生成模型
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0
#记录中间数值
train_acc_record=[]
train_loss_record=[]
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0

    # 训练
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)  # 返回概率最大的位置
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))
    train_acc_record.append(train_acc/len(train_set))
    train_loss_record.append(train_loss/len(train_loader))
#画出曲线
plot_curve(train_acc_record,title='train acc')
plot_curve(train_loss_record,title='train loss')
#保存模型
torch.save(model.state_dict(), './housing_question/housing_model.csv')
print('saving model at last epoch')
#创建用于测验的模型
model = Classifier().to(device)
model.load_state_dict(torch.load('./housing_question/housing_model.csv'))

predict = torch.tensor([])
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        predict=torch.cat([predict,outputs],axis=0)

predict = predict.numpy()[:,0:1]
test_final=np.append(predict,test_y)
test_final=test_final.reshape(2,56961).transpose()
test_final=test_final[np.argsort(test_final[:,0])]

TPR,FPR=AUC(test_final)
auc=0
for i in range(len(TPR)-1):
    auc+=TPR[i]*(FPR[i]-FPR[i+1])
print(auc)
plt.plot(FPR,TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
import glob
import os
from PIL import Image
import pandas as pd
import numpy as np

import piexif
import imghdr

#-------------------------------------超参数定义-------------------------------------

batch_size = 16 #一个batch的size
learning_rate = 0.01
num_epoches = 501 #总样本的迭代次数

#-------------------------------------数据集部分--------------------------------------

class MyDataset(Dataset):
    def __init__(self, path, Train=True, Len=-1, resize=-1, img_type='jpg', remove_exif=False, labelpath=''):
        all_label = pd.read_csv(labelpath)
        label = all_label.to_numpy()[:,1]
        for i in range(len(label)):
            if label[i] == -1:
                label[i] = 0
        label = label.astype(float)
        label = torch.from_numpy(label)
        self.label = F.one_hot(label.to(torch.int64),2)

        if resize != -1:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        img_format = '*.%s' % img_type

        if remove_exif:
            for name in glob.glob(os.path.join(path, img_format)):
                try:
                    piexif.remove(name)  # 去除exif
                except Exception:
                    continue

        # imghdr.what(img_path) 判断是否为损坏图片
        
        if Len == -1:
            self.dataset = [np.array(transform(Image.open(name).convert("RGB"))) for name in
                            glob.glob(os.path.join(path, img_format)) if imghdr.what(name)]
        else:
            self.dataset = [np.array(transform(Image.open(name).convert("RGB"))) for name in
                            glob.glob(os.path.join(path, img_format))[:Len] if imghdr.what(name)]
        self.dataset = np.array(self.dataset)
        self.dataset = torch.Tensor(self.dataset)
        self.Train = Train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx],self.label[idx]

#训练和测试集预处理
train_dataset = MyDataset(path=r'data\trainimages', resize=28, Len=2500, img_type='jpg',labelpath='data/train.csv')
val_dataset = MyDataset(path=r'data\valimages', resize=28, Len=500, img_type='jpg',labelpath='data/val.csv')
#加载数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#-------------------------------------网络部分--------------------------------------

class CNN(nn.Module):       
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding大小1
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

        self.fc1 = nn.Linear(50 * 50 * 16, 128)     # 第一个全连层，线性连接，输入节点数50×50×16，输出节点数128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):       # 重写父类forward方法，即前向计算，通过该方法获取网络输入数据后的输出值
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # n * 16 * 50 * 50

        x = x.view(x.size()[0], -1)     # 由于全连层输入的是一维张量，因此需要对输入的[50×50×16]格式数据排列成[40000×1]形式,n * 40000
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)      # 采用SoftMax方法将输出的2个输出值调整至[0.0, 1.0],两者和为1，并返回,tensor([[0.4544, 0.5456]], grad_fn=<SoftmaxBackward>)

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.feature_extr = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten()
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.feature_extr(x)
        x = self.classifier(x)
        return x

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=2) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.Softmax = nn.Softmax(1)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        # out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        out = self.Softmax(out)
        return out

class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA,self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        
        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16,24,kernel_size=5,padding=2)
    
        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24,24,kernel_size=3,padding=1)
        
        self.branch_pool = torch.nn.Conv2d(in_channels,24,kernel_size=1)
        
    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1,branch5x5,branch3x3,branch_pool]
        return torch.cat(outputs,dim=1)
class InceptionNet(torch.nn.Module):
    def __init__(self):
        super(InceptionNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88,20,kernel_size=5)
        
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        
        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408,2)
        self.softmax = nn.Softmax(1)
        
    def forward(self,x):
        # Flatten data from (n,1,28,28) to (n,784)
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size,-1)  # flatten
        x = self.fc(x)
        x = self.softmax(x)
        return x

#-------------------------------------选择模型--------------------------------------

# model = AlexNet()                 #224
# model = ResNet18(BasicBlock)      #224
# model = CNN()                     #200
model = InceptionNet()              #28
print(model)
if torch.cuda.is_available():
    model = model.cuda()

#-------------------------------------定义损失函数和优化器--------------------------------------

# criterion = nn.CrossEntropyLoss() 
criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#-------------------------------------开始训练-------------------------------------
print('Start Training!')
torch.cuda.empty_cache()
iter = 0 #迭代次数
for epoch in range(num_epoches):
    for img, label in train_loader:
        label = label.to(torch.float)        
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        out = model(img)
        # print(out,label)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter+=1
        if iter%200 == 0:
            print('epoch: {}, iter:{}, loss: {:.4}'.format(epoch, iter, loss.data.item()))

    if (epoch%50 == 0) & (epoch != 0):
        i = epoch/50
        # torch.save(model, 'AlexNet%03d.pth'% i)
        torch.save(model, 'Inception_%03d.pth'% i)
#-------------------------------------模型评估-------------------------------------
        print('Start eval!')
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for img, label in val_loader:
            label = label.to(torch.float)  
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data.item()*label.size(0)
            mask = (out == out.max(dim=1, keepdim=True)[0]).to(dtype=torch.float)
            num_correct = (mask == label).sum()/2
            eval_acc += num_correct.item()

        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_dataset)), eval_acc / (len(val_dataset))))





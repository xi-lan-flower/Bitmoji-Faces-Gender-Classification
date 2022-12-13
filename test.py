import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
import glob
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imghdr
#-------------------------------------超参数定义-------------------------------------
batch_size = 16 #一个batch的size
learning_rate = 0.01
num_epoches = 501 #总样本的迭代次数
def set_random_seed(seed, deterministic=False):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed(32767)
#-------------------------------------数据集部分--------------------------------------

class MyDataset(Dataset):
    def __init__(self, path, Train=True, Len=-1, resize=-1, img_type='jpg', remove_exif=False, labelpath=''):
        # all_label = pd.read_csv(labelpath)
        # label = all_label.to_numpy()[:,1]
        # for i in range(len(label)):
        #     if label[i] == -1:
        #         label[i] = 0
        # label = label.astype(float)
        # label = torch.from_numpy(label)
        # self.label = F.one_hot(label.to(torch.int64),2)

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
        return self.dataset[idx]

#训练和测试集预处理
test_dataset = MyDataset(path=r'data_new\end_testimages', resize=224, Len=1084, img_type='jpg')

#加载数据集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#-------------------------------------选择模型--------------------------------------
path="model/ResNet18_009.pth"
model = torch.load(path)
print(model)
if torch.cuda.is_available():
    model = model.cuda()

#-------------------------------------模型评估-------------------------------------
print('Start test!')
model.eval()

csvpath = "data_new/sample_submission.csv"
data = pd.read_csv(csvpath)
data = data.to_numpy()

index=0

for img in test_loader:
    if torch.cuda.is_available():
        img = img.cuda()
    out = model(img)
    for item in out:
        if item[0]>item[1]:
            data[index][1]=-1
        else:
            data[index][1]=1
        index+=1

print(data)

import csv
 
f = open('data.csv', 'w', encoding='utf-8', newline="")
csv_write = csv.writer(f)
csv_write.writerow(['image_id','is_male'])
for item in data:
    csv_write.writerow([item[0], item[1]])
f.close()



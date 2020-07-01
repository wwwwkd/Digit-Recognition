import torch


from torch import nn
from torch.nn import functional as F
from utils import Flatten



class LeNet5(nn.Module):
    '''
    1.Conv1 Block
    2.Conv2 Block
    3.Flatten layer1
    4.Flatten layer2
    ----参数设置------
    使用优化器adam：其中的 weight_decay = 0.01
    学习率learning rate：lr = 1e-4
    激活函数active function：relu
    '''
    def __init__(self):
        super(LeNet5, self).__init__() # 初始化函数并实现

        # -----Conv1 Block-----
        self.conv1_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.MaxPool2d(2)
        )

        # -----Conv2 Block-----
        self.conv2_block = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(2)
        )

        # -----Flatten layer1-----
        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(20*4*4, 120),  # (b,225*1*37)=>(b,4)
        )

        # -----Flatten layer2-----
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),  # (b,225*1*37)=>(b,4)
        )

        # -----Flatten layer3-----
        self.fc3 = nn.Sequential(
            Flatten(),
            nn.Linear(84, 10),  # (b,225*1*37)=>(b,4)
        )
    def forward(self, x): # 前向传播

        # [b, 1, 28, 28] => [b, 10, 12, 12]
        x1 = F.relu(self.conv1_block(x))
        #print('Conv1 Block', x1.shape)
        # [b, 10, 12, 12] => [b, 20, 4, 4]
        x2 = F.relu(self.conv2_block(x1))
        #print('Conv2 Block', x2.shape)
        #[b, 120*4*4] => [b, 120]
        x3 = self.fc1(x2)
        #print('flatetn layer1', x3.shape)
        # [b, 120] => [b, 84]
        x4 = self.fc2(x3)
        #print('flatetn layer1', x4.shape)
        # [b, 84] => [b, 10]
        x5 = self.fc3(x4)
        #print('flatetn layer1', x5.shape)

        return x5


def main():
    # test

    net = LeNet5()
    x = torch.randn(1, 1, 28, 28)
    a = net(x)
    print(a)



if __name__ == '__main__':
    main()
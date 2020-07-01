import torch
from matplotlib import pyplot as plt

from torch import nn


'''
    常用的几个辅助函数：
    
    1.Flatten函数：打平操作
    
    2.plot_curve函数：绘制loss曲线函数

'''

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item() #元素乘积
        return x.view(-1, shape)



def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

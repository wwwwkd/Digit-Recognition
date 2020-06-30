import torch
from torch import optim, nn


from torch.utils.data import DataLoader

from LeNet5 import LeNet5

from torchvision import datasets, transforms # 视觉工具包


batch_size=200 # 一次送入200张
learning_rate=0.01 # 学习率
epochs=40 # 最大迭代次数

device = torch.device('cuda') # 设备选择cuda
torch.manual_seed(1234) # 种子点

train_db = datasets.MNIST('../Digit Recognition/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
#文件名 train = true 代表下载的是那60k的训练数据 而不是剩下10k的test数据
#download = true 代表当前文件无数据则自动下载
#数据格式为numpy 将其转化成 tensor格式
#normalize是正则化的一个过程 由于图像的像素是0-1 只在0的右边 将其转换到0的两侧进行均匀分布 可以提高性能
#batch_size 代表一次加载多少张数据 shuffle 代表加载数据并打散
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)

test_db = datasets.MNIST('../Digit Recognition/data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])) # 加载测试集10k
test_loader = torch.utils.data.DataLoader(test_db,
    batch_size=batch_size, shuffle=True)


print('train:', len(train_db), 'test:', len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000]) # 将训练集在分为训练集50k和验证集10k
print('db1:', len(train_db), 'db2:', len(val_db))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True) #在进行数据打乱，数据增强
val_loader = torch.utils.data.DataLoader(
    val_db,
    batch_size=batch_size, shuffle=True)

def evalute(model, loader):
    '''

    测试函数：1.通过验证集评价网络最好得epoch
             2.通过测试集评价网络真实准确率

    注：均不用于反向传播调节网络，单纯评价
    '''
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x,y = x.to(device), y.to(device) # 改变设备类型 在GPU上进行运算
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
        # eq：pred与y进行比较相等的输出1 不同的输出0，然后sum累加，之后float转换成浮点数，最后item将tensor数据类型转换成numpy

    return correct / total # 返回平均准确率


def main():

    device = torch.device('cuda:0')
    model = LeNet5().to(device) # 加载网络模型
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.01) # 设置优化器参数 参数选择详见网络注释
    criteon = nn.CrossEntropyLoss().to(device) # 设置损失函数参数

    best_acc, best_epoch = 0, 0


    for epoch in range(epochs): # 训练

        for batch_idx, (data, target) in enumerate(train_loader):

            # data: [b, 1, 28, 28], target: [b]
            data, target = data.to(device), target.to(device)

            model.train() # 训练模型
            logits = model(data) # 获得网络输出
            loss = criteon(logits, target) # 根据损失函数得到loss

            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播，更新权重等参数信息，计算梯度
            optimizer.step() # 更新梯度，注：每一个step完成的是一个batchsize，每一个epoch完成的是一整个数据集
            if batch_idx % 100 == 0: # 每100个batchsize，输出一次迭代代数，已经训了训练集中多少数据，所占百分比，对应此时得loss
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader) # 每迭代一次，用validation_set进行验证
            print('Average_val_acc:', val_acc, 'epoch:', epoch)
            if val_acc> best_acc: # 用验证集来测试网络表现最好的代数 方便进行参数保存
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')



    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl')) # 加载在validation_set上表现最好时网络参数
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader) # 没有训练过的新图像即测试集来体现网络得真是性能
    print('test acc:', test_acc)

    # x = model(x)
    # pred = argmax(pred)




if __name__ == '__main__':
    main()

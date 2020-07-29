from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim

"""network"""
num_classes = 3
model = models.resnet101(pretrained=True)  # 预训练
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# 随机数种子
torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model = model.to(device)
    print('cuda:0')
else:
    print('cpu')


criterion = nn.CrossEntropyLoss()
# Stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 加速学习，0.9 - 10倍SDG算法
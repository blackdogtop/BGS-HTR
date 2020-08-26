import dataloader
import torch
from network import device, model
import matplotlib.pyplot as plt

img_loader = torch.utils.data.DataLoader(
    dataset = dataloader.ins_dataset_train,
    batch_size = 1, #一次迭代所使用的样本量
    shuffle = True,
    num_workers = 2,
)

for i, data in enumerate(img_loader, 0):
  images, labels = data

  images = images.to(device)
  labels = labels.to(device)

  if i == 0:
    img1 = images
  elif i == 1:
    img2 = images
  elif i == 2:
    img3 = images
  else:
    break

def layer_hook(self, input, output):
  plt.figure(figsize=(10, 10))
  for i in range(4):
    plt.subplot(4, 4, i+1)
    ft = output.data.clone().cpu()
    plt.imshow(ft[0,i,:,:,], cmap = 'gray')
    # print(ft[0,i:,:,].size())
  plt.show()


handle = model.conv1.register_forward_hook(layer_hook)
handle = model.layer2[0].register_forward_hook(layer_hook)
handle = model.layer3[0].register_forward_hook(layer_hook)
handle = model.layer4[2].register_forward_hook(layer_hook)

model(img1)
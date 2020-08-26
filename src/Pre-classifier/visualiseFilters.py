import math
import matplotlib.pyplot as plt
import numpy as np



def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors) #排序


def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]

def visualise_filter(conv):
  filter = conv.weight.data.cpu().numpy()
  out_channels = filter.shape[0]
  rows = np.min(get_grid_dim(out_channels))
  columns = np.max(get_grid_dim(out_channels))
  length = filter.shape[1]
  width = filter.shape[2]
  channels  = filter.shape[3]

  for i in range(1, 16):
    plt.subplot(4, 4, i)
    filter_im = np.zeros((length, width, channels))
    filter[i-1] = (filter[i-1] - np.mean(filter[i-1])) / np.std(filter[i-1]) #符合正态分布
    filter[i-1] = np.minimum(1, np.maximum(0, (filter[i-1] + 0.5)))
    # filter[i-1] = filter[i-1].transpose(1, 2, 0)#https://www.cnblogs.com/caizhou520/p/11227986.html

    filter_im = filter[i-1]
    plt.imshow(filter_im[:,:,-1], cmap='gray')#cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间 #https://www.cnblogs.com/denny402/p/5122594.html

# BGS-HTR


# 编译环境
操作系统 `macOS 10.15.5`, Python版本`3.7`
部分代码运行于`colab`

# 目前进度

>+ 18.05.2020 完成爬虫部分代码

>+ 15.06.2020 完成图片旋转

>+ 21.06.2020 完成图片分类

>+ 23.06.2020 完成图片分割（单元格分割）

>+ 09.07.2020 完成行分割（所有分割工作结束）

>+ 14.07.2020 提供标签

>+ 15.07.2020 word recognition

>+ 19.07.2020 校对标签

>+ 27.07.2020 text-line recognition

>+ 28.07.2020 pre-classifier

***

# 按功能部分讲解

### 下载BGS scanned records
> `downloader.ipynb` or `downloader.py`
* 根据url下载图片
* 获取某个BGS id下所有的图片(一个BGS id可能包含多个图片)
* 根据url获取某页下所有的图片
* 获取所有的区域(grid)的名字
* 获取某个区域(grid)的所有图片

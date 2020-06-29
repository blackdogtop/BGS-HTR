import numpy as np
import cv2
import math
import os

# 读取文件名
file_dir = '/Users/zzmacbookpro/Desktop/data/original'
file_names = []
for root, dirs, files in os.walk(file_dir):
    file_names = files
print(file_names)

def rotation(img):
    """
    统计图中长横线的斜率来判断整体需要旋转矫正的角度
    Rotate the image accoding to the overall slope of horizontal lines
    :params img: 初始图片
    :returns
        rotatedImg: 旋转之后的图片
    Modify:
        15.06.2020
    """

    image_copy = img.copy()

    # 灰度
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 边缘检测
    threshold1 = 50  # 最小阈值
    threshold2 = 150  # 最大阈值
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3)

    # 检测直线 - 统计概率霍夫线变换函数 输出检测到的直线的端点 (x0, y0, x1, y1)
    rho = 1  # 线段以像素为单位的距离精度,一般使用 1
    theta = np.pi / 180  # 线段以弧度为单位的角度精度,推荐用numpy.pi/180,一般使用 numpy.pi/180
    threshold = 100  # 累加平面的阈值参数,超过设定阈值才被检测出线段,值越大,检出的线段越长,检出的线段个数越少   (总像素长度)
    mLL = 200  # 线段以像素为单位的最小长度,组成一条直线的最少点的数量,点数量不足的直线将被抛弃          (局部像素长度)
    mLG = 10  # 同一方向上两条线段判定为一条线段的最大允许间隔(断裂),没超设定值则两条线段仍是两条线段    (单词间的gap,不要设定太大否则单词很容易被识别成直线)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=mLL, maxLineGap=mLG)
    if lines is None:  # 图像无直线,无法识别
        return img
    # for x1, y1, x2, y2 in lines[0]:
    #     cv2.line(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2_imshow(image_copy)

    pi = 3.1415
    theta_total = 0
    theta_count = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = math.atan(float(y2 - y1) / float(x2 - x1 + 0.001))  # 角度
        if theta < pi / 4 and theta > -pi / 4:
            theta_total = theta_total + theta
            theta_count += 1
            # cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)#添加直线在copy上
    if theta_count == 0:
        theta_count = 1
    theta_average = theta_total / theta_count  # 计算平均倾斜角度
    # print('旋转角度:{}'.format(theta_average*180/pi))

    # 旋转图片
    h, w, channels = img.shape
    img_change = cv2.getRotationMatrix2D((0, h), theta_average * 180 / pi, 1)  # 变换矩阵
    rotatedImg = cv2.warpAffine(img, img_change, (w, h), borderValue=(255, 255, 255))  # 旋转图片,白色填充
    # lineBasedImg = cv2.warpAffine(image_copy, img_change, (w, h), borderValue=(255,255,255)) #同时旋转原image和copy

    return rotatedImg

for i,file_name in enumerate(file_names):
    if file_name == '.DS_Store':
        continue
    image = cv2.imread('/Users/zzmacbookpro/Desktop/data/original/{}'.format(file_names[i]))
    rotatedImg = rotation(image)
    # cv2.imshow('image', image)
    # cv2.imshow('rotatedImg',rotatedImg)
    # cv2.waitKey(0)

    #保存图片到路径
    folder = os.path.exists('/Users/zzmacbookpro/Desktop/data/rotated')
    if not folder:
        os.makedirs('/Users/zzmacbookpro/Desktop/data/rotated')
    cv2.imwrite('/Users/zzmacbookpro/Desktop/data/rotated/{}'.format(file_names[i]), rotatedImg)
import cv2
import os
import numpy as np
import math
from PIL import Image

import segmentImg
import segmentLine

"""预处理移除图片横线和竖线"""


def getDarkPosition(image):
    """获取二值化图中非零像素点的坐标(x,y)"""

    p = []
    for i, x in enumerate(image):
        for j, y in enumerate(x):
            if y != 0:
                p.append((j, i))
    return p


def getFillValue(points, image):
    """取所有交点左上像素点的值累加并求平均值用于填充交点"""

    pixelValue = 0
    count = 0
    for p in points:
        x, y = p
        try:
            image[y - 1, x - 1]
        except IndexError:
            continue
        # 左上角的像素点不是已有的交点并且值小于128
        if (x-1,y-1) not in points and image[y - 1, x - 1][0] < 128:  # 128
            pixelValue += int(image[y - 1, x - 1][0])
            count += 1

    # print(count)
    return int(pixelValue/count) if count != 0 else 50


def getMorePoints(points):
    """将每个坐标点向上向下额外取一个像素"""
    morePoints = []
    for point in points:
        x, y = point
        morePoints.append((x, y))
        morePoints.append((x, y-1))
        morePoints.append((x, y+1))

    """indexError bug有待解决"""

    return morePoints


def rmUnderline(image):
    """移除下划线并填充字符与下划线交点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 获取水平线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))  # 50,1
    detected_hori_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_hori_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # 竖线
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)

    # 填充交点
    joints = cv2.bitwise_and(detected_hori_lines, dilatedrow)  # 交点
    points = getDarkPosition(joints)
    morePoints = getMorePoints(points)

    # 获取填充值并画点
    fillValue = getFillValue(points, image)
    # print(fillValue)

    # 删除横线并填充字符
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)  # 填充
    for p in morePoints:
        # cv2.circle(image, p, 1, (fillValue, fillValue, fillValue), 1)  # 使用此方法导致交点太粗
        x, y = p
        image[y, x] = [fillValue, fillValue, fillValue]

    return image


if __name__ == '__main__':
    path = '/Users/zzmacbookpro/Desktop/BGS-data-copy/line_segment/16288_13378153'
    files = os.listdir(path)
    savePath = '/Users/zzmacbookpro/Desktop/BGS-data-copy/preprocessed/' + path.split('/')[-1]
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    for file in files:
        if file == '.DS_Store':
            continue
        print(file)
        image = cv2.imread(path + '/' + file)
        # cv2.imshow('original', image)

        image = rmUnderline(image)  # 移除横线
        # image = segmentLine.rm_vertical_line(image)  # 移除竖线

        # cv2.imshow('rm',image)
        # cv2.waitKey(0)
        cv2.imwrite(savePath + '/{}'.format(file), image)

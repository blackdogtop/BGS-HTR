import cv2
import numpy as np
import math
import classify


def dots2line(img):
    """
    将虚线拟合为直线
    :params img:
    :returns img_copy:
    Modify:
        06.07.2020 - 添加异常处理
    """
    img_copy = img.copy()
    # 灰度
    gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 直线拟合
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=400, maxLineGap=20)
    # 异常处理
    if lines is None:
        return img_copy
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = math.atan(float(y2 - y1) / float(x2 - x1 + 0.001))
        if theta < np.pi / 4 and theta > -np.pi / 4:
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 0), 2)  # 黑色
    # cv2_imshow(img_copy)

    return img_copy


def fit_table(img):
    """
    拟合表格横线,竖线,外边框
    fit horizontal, vertical and outer border of a table
    :params img:
    :return img_copy
    Modify:
        20.06.2020
    """
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    h, w = binary.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # 横线
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))  # 竖线
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)

    joints = cv2.bitwise_and(dilatedcol, dilatedrow)  # 交点
    merge = cv2.add(dilatedcol, dilatedrow)
    # cv2_imshow(merge)

    contours, hierarchy = cv2.findContours(merge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy, contours, -1, (0, 0, 0), 3)  # 画轮廓
    length = len(contours)
    area_list = []

    for i in range(length):
        cnt = contours[i]
        area = cv2.contourArea(cnt)  # 面积
        area_list.append(area)
        # approx = cv2.approxPolyDP(cnt, 3, True)
        # x, y, w, h = cv2.boundingRect(approx)
        # cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 0), 3)

    # 找出最大面积的轮廓并画出(外轮廓)
    maxindex = area_list.index(max(area_list))
    cnt = contours[maxindex]
    approx = cv2.approxPolyDP(cnt, 3, True)  # 多边形拟合
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 0), 3)

    # cv2_imshow(img_copy)
    return img_copy


def table_out(img):
    """
    将图像二值化后识别横线与竖线,进而将每个单元格坐标提取出
    recognise table in the img
    :params img: 原始图片
    :return cell: (x, y, w, h) - 每个表格左上角的坐标,宽度,高度
                                 each coordinate of the upper left point of the rectangle, weight, height
    Modify:
        22.06.2020 - 调整参数
    """
    img_copy = img.copy()
    # 灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 二值化
    maxval = 255  # 当像素值超过了阈值(或小于),所赋予的值
    cv2Mean = cv2.ADAPTIVE_THRESH_MEAN_C  # 阈值取自相邻区域的平均值
    cv2Gaussian = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 阈值取值相邻区域的加权和,权重为一个高斯窗口
    thresh_type = cv2Mean  # 阈值的计算方法
    block_size = 15  # 图中分块大小
    binary = cv2.adaptiveThreshold(~gray, maxval, thresh_type, cv2.THRESH_BINARY, block_size, -2)
    rows, cols = binary.shape  # 高,宽

    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # MORPH_RECT-矩形, MORPH_CROSS-交叉, MORPH_ELLIPSE-椭圆
    eroded = cv2.erode(binary, kernel, iterations=1)  # 腐蚀
    dilatedcol = cv2.dilate(eroded, kernel, iterations=2)  # 膨胀
    # cv2_imshow(dilatedcol)

    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=2)
    # cv2_imshow(dilatedrow)

    # 全横线图与全竖线图叠加,并标识交点
    joints = cv2.bitwise_and(dilatedcol, dilatedrow)  # 对二进制数据进行"与"操作,即横线与竖线的交叉
    # cv2_imshow(joints)

    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    # cv2_imshow(merge)

    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(binary, merge)
    # cv2_imshow(merge2)

    # 根据矩形大小筛选矩形框，并画在矫正后的表格上
    cvExternal = cv2.RETR_EXTERNAL  # 只检测外轮廓
    cvList = cv2.RETR_LIST  # 检测的轮廓不建立等级关系
    cvCcomp = cv2.RETR_CCOMP  # 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息
    cvTree = cv2.RETR_TREE  # 建立一个等级树结构的轮廓
    cvMode = cvList  # 轮廓的检索模式

    cvNone = cv2.CHAIN_APPROX_NONE  # 存储所有的轮廓点，相邻的两个点的像素位置差不超过1
    cvSimple = cv2.CHAIN_APPROX_SIMPLE  # 压缩水平方向,垂直方向,对角线方向的元素,只保留该方向的终点坐标
    cvMethod = cvSimple  # 轮廓的近似办法

    contours, hierarchy = cv2.findContours(merge, cvMode, cvMethod)
    length = len(contours)
    cells = []  # 用于保存每个单元格

    for i in range(length):
        cnt = contours[i]
        area = cv2.contourArea(cnt)  # 计算轮廓的面积
        if area < 10:  # 排除过小的轮廓
            continue
        approx = cv2.approxPolyDP(cnt, 3, True)  # 多边形逼近
        x, y, w, h = cv2.boundingRect(approx)  # x,y是矩阵左上点的坐标, w,h是矩阵的宽和高
        rect = [x, y, w, h]
        roi = joints[y:y + h, x:x + w]  # region of interest
        joints_contours, joints_hierarchy = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if h > 20 and w > 20 and len(joints_contours) <= 50 and h < (rows - 50):  # 此处条件需要改
            # print(h)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255 - h * 3, h * 3, 0), 3)  # 轮廓越小越蓝,轮廓越大越绿
            cells.append(rect)

    # cv2_imshow(img_copy)
    # print(cells)

    return cells


def get_n_img_cell(image, coordinate_cells):
    """
    获取无表格图片(n)的每个cell
    :params image:
            coordinate_cells: 一个list包含所有cell的(x,y,w,h) - tableOut函数获取
    :return coordinate_cells: 处理之后的所有cell坐标
    Modify:
        23.06.2020
    """
    h, w, _ = image.shape

    x_list, y_list, w_list, h_list = [], [], [], []
    for i in coordinate_cells:
        x_list.append(i[0])
        y_list.append(i[1])
        w_list.append(i[2])
        h_list.append(i[3])

    # 将表头坐标添加到列表中
    max_h_index = h_list.index(max(h_list))
    max_y = coordinate_cells[max_h_index][1]
    header_img = image[0:max_y, :]
    coordinate_cells.append([0, 0, w, max_y])

    # 宽度大于(整张图-150)并且高度大于整张图三分之二则删除该轮廓
    for i, width in enumerate(w_list):
        if width > (w - 150) and coordinate_cells[i][-1] > h * 2 / 3:
            del coordinate_cells[i]

    x_list, y_list, w_list, h_list = [], [], [], []
    for i in coordinate_cells:
        x_list.append(i[0])
        y_list.append(i[1])
        w_list.append(i[2])
        h_list.append(i[3])

    # 如果一个cell在另一个cell之内则删除被包含的cell
    for i, xi in enumerate(x_list):
        for j, xj in enumerate(x_list[:i] + x_list[i + 1:]):
            xi1 = xi
            yi1 = y_list[i]
            xi2 = xi1 + w_list[i]
            yi2 = yi1 + h_list[i]

            xj1 = xj
            yj1 = (y_list[:i] + y_list[i + 1:])[j]
            xj2 = xj1 + (w_list[:i] + w_list[i + 1:])[j]
            yj2 = yj1 + (h_list[:i] + h_list[i + 1:])[j]

            if (xi1 > xj1) and (yi1 > yj1) and (xi2 < xj2) and (yi2 < yj2):
                try:
                    del coordinate_cells[i]
                except IndexError:
                    pass

    return coordinate_cells


def has_text(image, t):
    """
    根据轮廓个数判断是否有文本
    :params image:
            t: 图片类型
    :return True/False
    Modify:
        08.07.2020 - change tabular threshold from 5 to 4
    """
    if t == classify.image_type3:  # 表格
        rec = 4
        th = 4  # 5
    elif t == classify.image_type1:  # 虚线
        rec = 5
        th = 1
    elif t == classify.image_type2:  # 无表格
        rec = 3
        th = 5

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -80)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rec))
    eroded = cv2.erode(binary, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= th:
        return True
    else:
        return False


def has_multiple_text(height, h_th):
    """
    根据高度判断单元格内是否是多行文本
    :param height: 单元格高度
    :param h_th: 高度阀值
    :return: True/False
    Modify:
        08.07.2020
    """
    if height <= h_th:
        return False
    else:
        return True

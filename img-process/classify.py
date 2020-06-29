import os
import cv2

# 读取文件名
file_dir = '/Users/zzmacbookpro/Desktop/data/rotated'
file_names = []
for root, dirs, files in os.walk(file_dir):
    file_names = files
# print(file_names)

def rect_detect(img, th1=30, th2=120):
    """
    根据图像中轮廓个数与不同阀值比较判断图像类型 - 表格图(t),无表格图(n),虚线表格图(d)
    :params img:
            th1: 阀值1 用于区分虚线图和无表格图
            th2: 阀值2 用于区分无表格图和表格图
    :return type: -t / -n / -d
    Modify:
        21.06.2020
    """
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 15, -2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)) #横线
    eroded = cv2.erode(binary, kernel, iterations = 1)
    dilatedcol = cv2.dilate(eroded, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70)) #竖线
    eroded = cv2.erode(binary, kernel, iterations = 1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations = 1)
    merge = cv2.add(dilatedcol, dilatedrow)
    # cv2.imshow('merge',merge)
    # cv2.waitKey((0))
    contours, hierarchy = cv2.findContours(merge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < th1:
        return '-d'
    elif th1 <= len(contours) and len(contours) < th2:
        return '-n'
    else:
        return '-t'


for i, file_name in enumerate(file_names):
    if file_name == '.DS_Store':
        continue
    image = cv2.imread('/Users/zzmacbookpro/Desktop/data/rotated/{}'.format(file_names[i]))
    typeImg = rect_detect(image)
    file_name = file_name.split('.')[0] + typeImg + '.' + file_name.split('.')[1]

    # 保存图片
    folder = os.path.exists('/Users/zzmacbookpro/Desktop/data/classify')
    if not folder:
        os.makedirs('/Users/zzmacbookpro/Desktop/data/classify')
    cv2.imwrite('/Users/zzmacbookpro/Desktop/data/classify/{}'.format(file_name), image)
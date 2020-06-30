import cv2
import numpy as np
from matplotlib import pyplot as plt


def del_vertical_line(image):
    """
    删除竖直线
    :params
    :return
    Modify:
        27.06.2020
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 得到竖直线并删除
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected__ver_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    cnts = cv2.findContours(detected__ver_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    return image


def del_horizontal_line(image):
    """
    删除水平线
    :return:
    Modify:
        27.06.2020
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 得到水平线并删除
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detected_hori_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_hori_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    return image


def rm_noise(image):
    """
    移除噪点
    :return:
    Modify:
        28.06.2020
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1]  # get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 10:  # filter small dotted regions
            img2[labels == i + 1] = 255
    res = cv2.bitwise_not(img2)
    return res


def line_seg_regu(original_img, img_without_noise):
    """
    对于小图像的行分割 (需要传入灰度图)
    :param
    :return:
    Modify:
        28.06.2020 - 重命名
    """
    # gray = cv2.cvtColor(none_noise_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~img_without_noise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -80)
    kernel = np.ones((8, 1), np.uint8)  # 1,1
    img_erode = cv2.erode(binary, kernel, iterations=1)  #
    kernel = np.ones((1, 60), np.uint8)  # 2,40
    img_dilation = cv2.dilate(img_erode, kernel, iterations=1)
    # cv2.imshow('img_dilation', img_dilation)

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        area = cv2.contourArea(ctr)
        x, y, w, h = cv2.boundingRect(ctr)
        # 限定条件
        if area < 440:
            pass
            # continue
        if h < 7:
            pass
            # continue
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (90, 0, 255), 1)
        # print(area)
    if original_img.shape[0] > 1000:
        original_img = cv2.resize(original_img, (80, 800))
    # cv2.imshow('img_dilation',img_dilation)
    # cv2.imshow('original_img',original_img)
    # cv2.waitKey(0)
    return original_img


def line_seg_large(original_img, img_without_noise):
    """
    对于大图像的行分割
    :param
    :return:
    Modify:
        28.06.2020 - 重命名
    """
    original_cpy1 = original_img.copy()
    original_cpy2 = original_img.copy()

    # gray = cv2.cvtColor(none_noise_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~img_without_noise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -80)
    kernel = np.ones((2, 5), np.uint8)  # 1,5
    img_erode = cv2.erode(binary, kernel, iterations=1)  # 2
    kernel = np.ones((10, 150), np.uint8)  # 10,80
    img_dilation = cv2.dilate(img_erode, kernel, iterations=1)

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        area = cv2.contourArea(ctr)
        x, y, w, h = cv2.boundingRect(ctr)
        # 限定条件
        if area < 450:
            continue
        if h < 10 or w < 10:
            pass
            # continue
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.rectangle(original_cpy1, (x, y), (x + w, y + h), (0, 0, 0), -1) # 填充
        # 三元表达式防止越界
        y1 = (y - 10) if (y - 10) > 0 else 0
        y2 = (y + h + 15) if (y + h + 15) < original_cpy1.shape[0] else original_cpy1.shape[0]
        # roi - 竖直方向上下多提取10,15个像素
        roi = original_cpy1[y1:y2, x:x + w]
        cv2.imshow('roi', original_img[y1:y2, x:x + w])
        connectedComponent(roi)
        # 重新赋值以免之前绘制的矩形影响识别
        original_cpy1 = original_cpy2

    cv2.imshow('img_dilation', img_dilation)
    cv2.imshow('original_img', original_img)
    cv2.waitKey(0)
    return original_img


def connectedComponent(roi):
    """
    寻找roi连接区域
    :param roi
    :return:
    Modify:
        30.06.2020
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(~gray, 127, 255, cv2.THRESH_BINARY)[1]
    ret, labels = cv2.connectedComponents(binary)
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue == 0] = 0
    # show image
    cv2.imshow('labeled.png', labeled_img)

    # 找到非黑像素数最多的connected component
    max_white_pixel = 0
    for label in range(1, ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        mask_sum = np.sum(mask)
        if mask_sum > max_white_pixel:
            max_white_pixel = mask_sum
            max_pixel_index = label  # 下标
    max_mask = np.array(labels, dtype=np.uint8)
    max_mask[labels == max_pixel_index] = 255
    cv2.imshow('max_mask', ~max_mask)
    cv2.waitKey(0)


# 读取文件名
img_names = []
with open('/Users/zzmacbookpro/Desktop/data/segment/cell.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        text_type = line.split()[1]
        if text_type == 'MultipleText':
            img_name = line.split()[0]
            img_names.append(img_name)

# 读取图片进行行分割
count = 0
for img_name in img_names:
    img_dir = img_name.split('-')[0] + '-' + img_name.split('-')[1]
    file = '/Users/zzmacbookpro/Desktop/data/segment/' + img_dir + '/' + img_name
    image = cv2.imread(file)
    # 保存图片
    # cv2.imwrite('/Users/zzmacbookpro/Desktop/multipletext/{}'.format(img_name),image)
    if (image.shape[0] > 1000 and image.shape[1] > 200) or (image.shape[1] > 600 and image.shape[0] > 100):  # 大图（难分割）
        pass
        count += 1
        print(img_name)
        img_cpy = image.copy()
        image = del_vertical_line(image)
        image = del_horizontal_line(image)
        img_without_noise = rm_noise(image)
        line_img = line_seg_large(img_cpy, img_without_noise)

    else:
        pass
        # count += 1
        # print(img_name)
        # img_cpy = image.copy()
        # image = del_vertical_line(image)
        # image = del_horizontal_line(image)
        # img_without_noise = rm_noise(image)
        # line_img = line_seg_regu(img_cpy,img_without_noise)
        # cv2.imshow('line_img',line_img)
        # cv2.waitKey(0)
        # 保存图片
        # cv2.imwrite('/Users/zzmacbookpro/Desktop/multipletext-segment/{}'.format(img_name),line_img)
print(count)

# 测试单个图片
# file = '/Users/zzmacbookpro/Desktop/segmentation/10612_10015680-t/10612_10015680-t-114.png'
# image = cv2.imread(file)
# img_cpy = image.copy()
# image = del_vertical_line(image)
# image = del_horizontal_line(image)
# img_without_noise = rm_noise(image)
# line_img = line_seg_regu(img_cpy,img_without_noise)

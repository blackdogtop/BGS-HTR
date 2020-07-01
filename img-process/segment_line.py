import cv2
import numpy as np
import copy
import watershed_line_segment

def rm_vertical_line(image):
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


def rm_horizontal_line(image):
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
            original_img: 原图
            img_without_noise: 经过去噪点处理的灰色二值图
    :return:
    Modify:
        01.07.2020
    """
    original_cpy1 = original_img.copy()
    original_cpy2 = original_img.copy()

    binary = cv2.adaptiveThreshold(~img_without_noise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -80)
    # 先腐蚀后膨胀
    # kernel = np.ones((5, 2), np.uint8)  # 1,5
    # img_erode = cv2.erode(binary, kernel, iterations=1)  # 2
    # kernel = np.ones((3, 100), np.uint8)  # 10,80
    # img_dilation = cv2.dilate(img_erode, kernel, iterations=1)
    # cv2.imshow('img_dilation', img_dilation)

    # 膨胀-腐蚀-膨胀
    kernel = np.ones((1, 200), np.uint8) #1, 200
    img_dilation = cv2.dilate(binary, kernel, iterations=1)
    kernel = np.ones((5, 300), np.uint8) #5, 300
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    kernel = np.ones((1, 400), np.uint8) #1, 400
    img_dilation = cv2.dilate(img_erode, kernel, iterations=1)
    cv2.imshow('img_dilation', img_dilation)

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        area = cv2.contourArea(ctr)
        x, y, w, h = cv2.boundingRect(ctr)
        # 限定条件
        if area < 450:
            pass
            continue
        if h < 7 or w < 10: # 8
            pass
            continue
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # 填充
        cv2.rectangle(original_cpy1, (x, y), (x + w, y + h), (0, 0, 0), -1)
        # 三元表达式防止越界
        y_minus, y_plus = 15, 15 # 竖直方向上下各多提取10, 15个像素
        y1 = (y - y_minus) if (y - y_minus) > 0 else 0
        y2 = (y + h + y_plus) if (y + h + y_plus) < original_cpy1.shape[0] else original_cpy1.shape[0]
        roi = original_cpy1[y1:y2, x:x + w]
        # cv2.imshow('roi', original_img[y1:y2, x:x + w])
        roi_coordinate = (y_minus, w, h)
        contours_coordinate = connectedComponent(roi, roi_coordinate)
        # 深拷贝以免填充的矩形影响识别
        original_cpy1 = copy.deepcopy(original_cpy2)
    cv2.imshow('original_img with contours', original_img)
    cv2.waitKey(0)
    return original_img


def connectedComponent(roi, roi_coordinate):
    """
    寻找roi连接区域
    :param roi:
           roi_coordinate: 向上多提取的像素数, 宽, 高
    :return contours_coordinate_list: 每个connected component的轮廓坐标
    Modify:
        01.07.2020
    """
    # 读取坐标
    x, y, w, h = 0, roi_coordinate[0], roi_coordinate[1], roi_coordinate[2]

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
    # cv2.imshow('labeled.png', labeled_img)

    # 找到白色像素数最多的connected component
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
    # convert to RGB
    max_mask = cv2.cvtColor(max_mask, cv2.COLOR_GRAY2RGB)
    # cv2.imshow('max_mask', max_mask)
    # cv2.waitKey(0)

    # 主体轮廓填充
    cv2.rectangle(max_mask, (x, y), (x + w, y + h), (0, 0, 0), -1)
    # cv2.imshow('dark fill', max_mask)
    # cv2.waitKey(0)

    # 转换灰度二值
    gray_mask = cv2.cvtColor(max_mask, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((2, 2), np.uint8)
    img_dilation = cv2.dilate(binary_mask, kernel, iterations=1) # 膨胀

    # 轮廓提取
    ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    contours_coordinate_list = []
    for i, ctr in enumerate(sorted_ctrs):
        x_component, y_component, w_component, h_component = cv2.boundingRect(ctr)
        contours_coordinate_list.append([x_component, y_component, w_component, h_component])
        cv2.rectangle(roi, (x_component, y_component), (x_component + w_component, y_component + h_component), (0, 255, 0), 1)
    # cv2.imshow('component contours', roi)
    # cv2.waitKey(0)

    return contours_coordinate_list







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
        if image.shape[0] > 197 and image.shape[1] > 1000: # 表头
            pass
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
        elif 180 < image.shape[0] and image.shape[0] < 200 and 630 < image.shape[1] and image.shape[1] < 700: # 小表头
            pass
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
        else: # 多行文本
            pass
            count += 1
            print(img_name)
            img_cpy = image.copy()
            image = rm_vertical_line(image)
            image = rm_horizontal_line(image)
            img_without_noise = rm_noise(image)
            line_img = line_seg_large(img_cpy, img_without_noise)

            # # 分水岭方法 - 表现不佳
            # line_seg = watershed_line_segment.WatershedLineSegment(image)
            # # image = rm_vertical_line(image)
            # img_line_erode = line_seg.get_h_projection()
            # connected_img = line_seg.connectComponents(img_line_erode)
            # line_seg.Watershed(connected_img)
    else:
        pass
        # count += 1
        # print(img_name)
        # img_cpy = image.copy()
        # image = rm_vertical_line(image)
        # image = rm_horizontal_line(image)
        # img_without_noise = rm_noise(image)
        # line_img = line_seg_regu(img_cpy,img_without_noise)
        # cv2.imshow('line_img',line_img)
        # cv2.waitKey(0)
        # 保存图片
        # cv2.imwrite('/Users/zzmacbookpro/Desktop/multipletext-segment/{}'.format(img_name),line_img)


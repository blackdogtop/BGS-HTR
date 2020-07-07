import cv2
import numpy as np
import copy
import watershed_line_segment


def rm_vertical_line(image):
    """
    检测竖直线并用白色填充
    :param image: 一张RGB图片
    :return image: 将竖直线白色填充后的图片
    Modify:
        27.06.2020
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 获取竖直线
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected_ver_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts = cv2.findContours(detected_ver_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)  # 填充

    return image


def rm_horizontal_line(image):
    """
    检测水平线并用白色填充
    :param image: 一张RGB图片
    :return image: 将水平线白色填充后的图片
    Modify:
        05.07.2020
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 获取水平线并删除
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))  # 50,1
    detected_hori_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_hori_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)  # 填充

    return image


def rm_noise(image):
    """
    移除噪点
    :param image: 一张RGB图片
    :return: 移除噪点后的二值化图片
    Modify:
        28.06.2020
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

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
    将传入的灰度二值小图像进行行分割
    :param original_img: 原图
           img_without_noise: 经过去噪点处理的灰度二值图
    :return:
    Modify:
        28.06.2020
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
    # if original_img.shape[0] > 1000:
    #     original_img = cv2.resize(original_img, (80, 800))
    # cv2.imshow('img_dilation',img_dilation)
    # cv2.imshow('original_img',original_img)
    # cv2.waitKey(0)
    return original_img


def line_segment(original_img, binary_img):
    """
    将传入的灰度二值多行文本图像进行行分割
    :param
            original_img: 一张RBG图片
            binary_img: 二值化图片(去噪)
    :return:
    Modify:
        07.07.2020
    """
    original_cpy1 = original_img.copy()
    original_cpy2 = original_img.copy()

    binary = cv2.adaptiveThreshold(~binary_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -80)
    # 膨胀-腐蚀-膨胀
    kernel = np.ones((5, 20), np.uint8)  # 1,200
    img_dilation = cv2.dilate(binary, kernel, iterations=1)
    kernel = np.ones((3, 40), np.uint8)  # 5,300
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    kernel = np.ones((2, 70), np.uint8)  # 1,200
    img_dilation = cv2.dilate(img_erode, kernel, iterations=1)
    # cv2.imshow('img_dilation', img_dilation)

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        area = cv2.contourArea(ctr)
        x, y, w, h = cv2.boundingRect(ctr)
        main_ctr = [x, y, w, h]
        if area < 1250:  # 450
            continue
        if h < 10 or w < 10:  # 8
            continue
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        """连通区域获取"""
        # 填充
        cv2.rectangle(original_cpy1, (x, y), (x + w, y + h), (0, 0, 0), -1)
        y_minus, y_plus = 15, 15  # 竖直方向上下各多提取15, 15个像素
        # 三元表达式防止越界
        y1 = (y - y_minus) if (y - y_minus) > 0 else 0
        y2 = (y + h + y_plus) if (y + h + y_plus) < original_cpy1.shape[0] else original_cpy1.shape[0]
        y_extra = [y1, y2]
        roi = original_cpy1[y1:y2, x:x + w]
        roi_coordinate = (y_minus, w, h)
        # 获取连接区域
        contours_coordinate = get_connected_component(roi, roi_coordinate)
        # 每个connected component画矩形边框
        cc_ctrs = []  # 用于存放每个连通区域在整张图上的坐标
        for cc_ctr in contours_coordinate:
            cc_x, cc_y, cc_w, cc_h = (x + cc_ctr[0]), (y - y_minus) + cc_ctr[1], cc_ctr[2], cc_ctr[3]
            cv2.rectangle(original_img, (cc_x, cc_y), (cc_x + cc_w, cc_y + cc_h), (0, 255, 0), 1)
            cc_ctrs.append([cc_x, cc_y, cc_w, cc_h])
            # cv2.imshow('cc',original_img)
            # cv2.waitKey(0)
        # 深拷贝以免填充的矩形影响识别
        original_cpy1 = copy.deepcopy(original_cpy2)

        """获取行分割图"""
        line_seg_img = get_line_segment_image(original_cpy2, main_ctr, y_extra, cc_ctrs)
        # cv2.imshow('original_cpy2',original_cpy2)
        # cv2.imshow('line_seg_img',line_seg_img)
        # cv2.waitKey(0)

    cv2.imshow('original_img with contours', original_img)
    cv2.waitKey(0)


def get_connected_component(roi, roi_coordinate):
    """
    获取roi连通区域
    :param roi: 一个RGB区域
           roi_coordinate: roi相较于主体轮廓向上多提取的像素数, 宽, 高
    :return contours_coordinate_list: 每个连通区域坐标
    Modify:
        01.07.2020
    """
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
    img_dilation = cv2.dilate(binary_mask, kernel, iterations=1)  # 膨胀

    # 轮廓提取
    ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    contours_coordinate_list = []
    for i, ctr in enumerate(sorted_ctrs):
        x_component, y_component, w_component, h_component = cv2.boundingRect(ctr)
        contours_coordinate_list.append([x_component, y_component, w_component, h_component])
        cv2.rectangle(roi, (x_component, y_component), (x_component + w_component, y_component + h_component),
                      (0, 255, 0), 1)
    # cv2.imshow('component contours', roi)
    # cv2.waitKey(0)

    return contours_coordinate_list


def get_line_segment_image(image, main_ctr, y_extra, cc_ctrs):
    """
    将每行主体轮廓和额外的连通区域保留，其余填充白色，获取行分割图片
    :param image: 原图
    :param main_ctr: 主体轮廓坐标
    :param y_extra: 竖直向上向下额外提取的区域坐标
    :param cc_ctrs: 连通区域在整张图上的坐标
    :return line_seg_img: 分割之后的行图片
    Modify:
        07.07.2020
    """
    x, y, w, h = main_ctr
    y = y + 1  # 多提取1防止像素断裂
    y_top, y_bottom = y_extra

    # 白色背景图
    white_bg = 255 * np.ones_like(image)
    # 主体轮廓
    main_roi = image[y:y+h, x:x+w]
    white_bg[y:y+h, x:x+w] = main_roi
    # 连通区域
    for cc_ctr in cc_ctrs:
        cc_x, cc_y, cc_w, cc_h = cc_ctr
        cc_roi = image[cc_y:cc_y+cc_h, cc_x:cc_x+cc_w]
        white_bg[cc_y:cc_y+cc_h, cc_x:cc_x+cc_w] = cc_roi

    white_bg = white_bg[y_top:y_bottom, x:x+w]
    return white_bg





# 读取文件名
img_names = []
with open('/Users/zzmacbookpro/PycharmProjects/img_process/data/segmented/cell.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        text_type = line.split()[1]
        if text_type == 'MultipleText':
            img_name = line.split()[0]
            img_names.append(img_name)

# 读取图片进行行分割
for img_name in img_names:
    img_dir = img_name.split('-')[0]
    file = '/Users/zzmacbookpro/PycharmProjects/img_process/data/segmented/' + img_dir + '/' + img_name
    image = cv2.imread(file)
    if (image.shape[0] > 1000 and image.shape[1] > 200) or (image.shape[1] > 600 and image.shape[0] > 100):
        if 197 < image.shape[0] < 500 and image.shape[1] > 1000:  # 直接忽略表头
            pass
        elif 180 < image.shape[0] < 200 and 630 < image.shape[1] < 700:  # 小表头
            pass
            # print(img_name)
            # img_cpy = image.copy()
            # image = rm_vertical_line(image)
            # image = rm_horizontal_line(image)
            # image = rm_noise(image)
            # line_segment(img_cpy, image)
        else:  # 多行文本
            pass
            # print(img_name)
            # img_cpy = image.copy()
            # image = rm_vertical_line(image)
            # image = rm_horizontal_line(image)
            # image = rm_noise(image)
            # line_segment(img_cpy, image)

            # 使用分水岭方法行分割
            # line_seg = watershed_line_segment.WatershedLineSegment(image)
            # image = rm_vertical_line(image)
            # img_line_erode = line_seg.get_h_projection()
            # connected_img = line_seg.connectComponents(img_line_erode)
            # line_seg.Watershed(connected_img)
    else:
        pass
        # print(img_name)
        # img_cpy = image.copy()
        # image = rm_vertical_line(image)
        # image = rm_horizontal_line(image)
        # image = rm_noise(image)
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # line_img = line_seg_regu(img_cpy,image)
        # cv2.imshow('line_img',line_img)
        # cv2.waitKey(0)

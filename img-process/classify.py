import cv2


def ctr_detect(img, th1=30, th2=100):
    """
    根据图像中轮廓个数与不同阀值比较判断图像类型 - 表格图(t),无表格图(n),虚线表格图(d)
    :params img:
            th1: 阀值1 用于区分虚线图和无表格图
            th2: 阀值2 用于区分无表格图和表格图
    :return type: tabular / non-tabular / dashed-line
    Modify:
        21.06.2020
    """
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # 横线 - 40,1
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_col = cv2.dilate(eroded, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))  # 竖线 - 1,70
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_row = cv2.dilate(eroded, kernel, iterations=1)
    merge = cv2.add(dilated_col, dilated_row)
    # cv2.imshow('merge',merge)
    # cv2.waitKey((0))
    contours, hierarchy = cv2.findContours(merge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < th1:
        return 'dashed-line'
    elif th1 <= len(contours) < th2:
        return 'non-tabular'
    else:
        return 'tabular'

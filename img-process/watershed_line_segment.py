import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np

class WatershedLineSegment():
    """
    使用分水岭方法行分割
    """
    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰度
        _, self.thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # 二值化
        self.height, self.weight, _ = image.shape # 高，宽

        # 定义两个kernel
        self.kernel1 = np.ones((3, 3), np.uint8)
        self.kernel2 = np.ones((5, 20), np.uint8) # 5,500
        # self.kernel3 = np.ones((5, int(self.height / 400) * 100), np.uint8)
        self.kernel3 = np.ones((3, 40), np.uint8)
        self.kernel4 = np.ones((2, 70), np.uint8)

    def get_h_projection(self):
        """
        获取水平投影
        :return img_erode:
        Modify:
            01.07.2020
        """
        img_dilate = cv2.dilate(self.thresh, self.kernel2, iterations=1) # 横向膨胀
        img_erode = cv2.erode(img_dilate, self.kernel3, iterations=1)    # 2 - 横向腐蚀
        img_dilate = cv2.dilate(img_erode, self.kernel4, iterations=1)  # 横向膨胀

        # display
        # cv2.imshow('get_h_projection', img_dilate)

        return img_dilate

    def connectComponents(self, image):
        """
        连通区域获取
        :return image:
        Modify:
            01.07.2020
        """
        _, marked = cv2.connectedComponents(image, connectivity=4)

        # 将每个连通区域对应一个颜色
        label_hue = np.uint8(179 * marked / np.max(marked))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # RGB转换
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0

        # 腐蚀且灰度
        labeled_img = cv2.erode(labeled_img, self.kernel1)
        image = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

        # display
        # cv2.imshow('connectComponents', image)
        # cv2.waitKey(0)

        return image

    def Watershed(self, image):
        """
        分水岭
        :param image:
        :return:
        Modify:
            01.07.2020
        """

        original_img = self.image.copy()

        # Finding Eucledian distance to determine markers
        temp = ndimage.distance_transform_edt(image)  # 计算图像中像素点到最近零像素点的距离
        Max = peak_local_max(temp, indices=False, min_distance=5, labels=image)  # 30, 局部峰值坐标
        markers = ndimage.label(Max, structure=np.ones((3, 3)))[0]

        # Applying Watershed to the image and markers
        res = watershed(temp, markers, mask=image)

        # If the elements are same, they belong to the same object (Line)
        for i in np.unique(res):
            if i == 0:
                continue
            mask = np.zeros(self.gray.shape, dtype="uint8")
            mask[res == i] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)

            # Drawing a rectangle around the object
            [x, y, w, h] = cv2.boundingRect(c)
            if w > 5 and h > 5:
                cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('contours img', original_img)
        cv2.waitKey(0)



import cv2
import segment_line
import os
import math
import numpy as np


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry: entry[0][0])


def prepareImg(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel


def main():
    """reads images from data/ and outputs the word-segmentation to out/"""

    data_dir = './data/'
    line_seg_dir = data_dir + 'line_segment'  # 行分割目录
    word_seg_dir = data_dir + 'word_segment'

    files = os.listdir(line_seg_dir)
    for file in files:
        if file == '.DS_Store' or file == 'lines.txt':
            continue
        img_dir = line_seg_dir + '/' + file
        images_path = os.listdir(img_dir)
        for image_path in images_path:
            image_name = image_path.split('.')[0]
            image_path = img_dir + '/' + image_path
            image = cv2.imread(image_path)
            image = segment_line.rm_horizontal_line(image)  # 移除横线

            # prepare image by resizing it to fixed height and converting it to grayscale
            img = prepareImg(image, 50)
            # execute segmentation with given parameters
            # -kernelSize: size of filter kernel (odd integer)
            # -sigma: standard deviation of Gaussian function used for filter kernel
            # -theta: approximated width/height ratio of words, filter function is distorted by this factor
            # - minArea: ignore word candidates smaller than specified area
            res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)

            # write output to 'out/inputFileName' directory
            if not os.path.exists(word_seg_dir + '/' + image_name.split('-')[0] + '/' + image_name):
                os.makedirs(word_seg_dir + '/' + image_name.split('-')[0] + '/' + image_name)

            # iterate over all segmented words
            print('Segmented into %d words' % len(res))
            for (j, w) in enumerate(res):
                (wordBox, wordImg) = w
                (x, y, w, h) = wordBox
                cv2.imwrite(word_seg_dir + '/' + image_name.split('-')[0] + '/' + image_name + '/'+ '{}'.format(j) + '.png', wordImg)
                # cv2.imwrite('../out/%s/%d.png' % (f, j), wordImg)  # save word
                cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)  # draw bounding box in summary image

            # output summary image with bounding boxes around words
            cv2.imwrite(word_seg_dir + '/' + image_name.split('-')[0] + '/' + image_name + '/summary.png', img)



if __name__ == '__main__':
	main()
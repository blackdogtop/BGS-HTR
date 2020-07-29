from handwritten_api import HandWritten
import cv2


class FilePaths:
    imageName = '16288_13378153-043.png'
    file = '/Users/zzmacbookpro/Desktop/BGS-data-copy/preprocessed/' + imageName.split('-')[0] + '/' + imageName


def main():
    detector = HandWritten()
    text = detector.detect(img_path=FilePaths.file)
    return text


if __name__ == '__main__':
    text = main()
    print('Recognized: ' + text)
    image = cv2.imread(FilePaths.file)
    cv2.imshow('test', image)
    cv2.waitKey(0)

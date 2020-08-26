from handwritten_api import HandWritten
import cv2


class FilePaths:
    imageName = '9637_10014179-111.png'
    file = '../../data/line-segment/' + imageName.split('-')[0] + '/' + imageName
    groundTruthFile = '../../data/lines-v2.txt'
    predictFile = '../../data/lineHTR-prediction.txt'
    imagePath = '../../data/line-segment/'
    # predictFile = '../../data/lineHTR-prediction-preprocess.txt'
    # imagePath = '../../data/line-segment-preprocessed/'


def main():
    """recognise a single test image"""
    # detector = HandWritten()
    # text = detector.detect(img_path=FilePaths.file)
    # print(text)

    """recognise all BGS segmented text-lines"""
    detector = HandWritten()
    with open(FilePaths.groundTruthFile, 'r') as f:
        lines = f.readlines()

    with open(FilePaths.predictFile, 'w') as f:
        for line in lines:
            if line[0] == '#':
                continue
            image_name = line.split(' ')[0].split('-')[0]
            line_name = line.split(' ')[0]
            if line.split()[2] != 'handwritten':  # 仅识别手写文本
                f.write(line_name + '\n')
                continue
            prediction = detector.detect(img_path=FilePaths.imagePath + image_name + '/' + line_name)
            prediction = prediction.split(' ')
            prediction = '|'.join(prediction)
            f.write(line_name + ' ')
            f.write(prediction + '\n')


if __name__ == '__main__':
    # text = main()
    # print('Recognized: ' + text)
    # image = cv2.imread(FilePaths.file)
    # cv2.imshow('test', image)
    # cv2.waitKey(0)

    main()

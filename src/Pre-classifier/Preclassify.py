from dataloader import FilePaths, test_loader, test_data_df, test_paths
import dataloader
import torch
import cv2
import os

"""测试"""


class SavedPath:
    savedRootPath = '../../data/'
    resnet101Pretrained = savedRootPath + 'preclassifier-prediction.txt'
    resnet101PretrainedLabel = '# preclassifier - ResNet 101 pretrained prediction on BGS text-lines dataset'


def sortDictKey(dict):
    return [(k, dict[k]) for k in sorted(dict.keys())]


def main():
    model = torch.load(FilePaths.savedModel, map_location=torch.device('cpu'))
    model.eval()

    results = []
    with torch.no_grad():
        # Iterate over the test set
        for data in test_loader:
            images, labels = data

            outputs = model(images)

            # torch.max is an argmax operation
            _, predicted = torch.max(outputs.data, 1)

            for p in predicted:
                results.append(p.item())

    return results


if __name__ == '__main__':
    # 获取预分类器预测结果
    predictedResults = {}
    results = main()
    for i, path in enumerate(test_data_df['path']):
        imageName = path.split('/')[-1]
        predictedResults[imageName] = results[i]
    # print(predictedResults)
    sortedPreclassify = sortDictKey(predictedResults)  # 根据文件名排序
    # print(sortedPreclassify)
    # 写入txt文件
    with open(SavedPath.resnet101Pretrained, 'w') as f:
        for i, predictedLine in enumerate(sortedPreclassify):
            segmentName = predictedLine[0]
            predict = predictedLine[1]
            # print(segmentName, predict)
            if predict == 0:
                imageType = 'digital'
            elif predict == 1:
                imageType = 'handwritten'
            elif predict == 2:
                imageType = 'printed'
            f.write(segmentName)
            f.write(' ' + imageType)
            if i != len(sortedPreclassify):
                f.write('\n')

    # for i, path in enumerate(test_data_df['path']):
    #     fileName = path.split('/')[-1]
    #     # 获取未提供标签的BGS文本行文件
    #     if fileName == 'lines.txt':
    #         with open(path, 'r') as f:
    #             BGSLines = f.readlines()
    #         # print(BGSLines)
    #         for j, BGSline in enumerate(BGSLines):
    #             segmentName = BGSline.split()[0]
    #             imageName = segmentName.split('-')[0]
    #             segmentCoordinate = '(' + BGSline.strip().split('(')[1]
    #             # print(imageName)
    #             # 判断正在读取的segmentation的图片名是否等于上一个segmentation的图片名
    #             if imageName != BGSLines[j-1].split()[0].split('-')[0]:
    #                 print(imageName)
    # preclassifyDict = {}
    # results = main()
    # for i, path in enumerate(test_data_df['path']):
    #     imageName = path.split('/')[-1]
    #     preclassifyDict[imageName] = results[i]
    #     # print(imageName, results[i])
    # print(preclassifyDict)
    #
    # sortedPreclassify = sortDictKey(preclassifyDict)

    # for i, path in enumerate(test_data_df['path']):
    #     imageName = path.split('/')[-1]
    #     image = cv2.imread(path)
    #     # print(results[i])
    #     print('image {} is:'.format(imageName), end=' ')
    #     if results[i] == 0:
    #         print('digital')
    #     elif results[i] == 1:
    #         print('handwritten')
    #     else:
    #         print('printed')
    #     cv2.imshow('{}'.format(imageName), image)
    #     cv2.waitKey(0)

from dataloader import FilePaths, test_loader, test_data_df, test_paths
import dataloader
import torch
import cv2
import os

"""测试"""


class SavedPath:
    savedRootPath = '../../data/'
    resnet101Pretrained =        savedRootPath + 'preclassifier-prediction.txt'
    resnet101WithoutPretrained = savedRootPath + 'preclassifier-prediction-101.txt'
    resnet50Pretrained =         savedRootPath + 'preclassifier-prediction-50-pretrained.txt'
    resnet50WithoutPretrained =  savedRootPath + 'preclassifier-prediction-50.txt'
    vgg19Pretrained =            savedRootPath + 'preclassifier-prediction-19-pretrained.txt'
    vgg19WithoutPretrained =     savedRootPath + 'preclassifier-prediction-19.txt'



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
    with open(SavedPath.vgg19Pretrained, 'w') as f:
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

"""compare generated prediction file with ground truth labels file"""

class FilePaths:
    labelledFile = '../../data/lines-v2.txt'
    preclassifyFile = '../../data/preclassifier-prediction.txt'

def sortDictKey(dict):
    """字典根据key重新排序"""
    return [(k, dict[k]) for k in sorted(dict.keys())]

def comparePreclassify(file1, file2):
    """比较预分类器结果"""
    def writeInDict(lines, targetDict, p):
        # 写入文本行到字典
        for line in lines:
            if line[0] == '#':
                continue
            segmentName = line.split()[0]
            column = line.strip().split()[p]  # 指定写入什么column
            targetDict[segmentName] = column
        return targetDict

    # 获取行文本
    with open(file1, 'r') as f:
        lines1 = f.readlines()
    with open(file2, 'r') as f:
        lines2 = f.readlines()
    # 字典存储
    line1Dict = {}
    line2Dict = {}
    line1Dict = writeInDict(lines1, line1Dict, 1)
    line2Dict = writeInDict(lines2, line2Dict, 2)
    # 根据key(segment name)重新排序
    orderedLine1 = sortDictKey(line1Dict)
    orderedLine2 = sortDictKey(line2Dict)
    # 比较文件长度
    if len(orderedLine1) != len(orderedLine2):
        return 'The length of two files are different'
    # 比较文件顺序是否一致
    for i, l1 in enumerate(orderedLine1):
        segmentName1 = l1[0]
        if segmentName1 != orderedLine2[i][0]:
            print(segmentName1, orderedLine2[i][0])
            return 'The reordered two files are different'
    # 比较生成的预测和标签(非none)计算准确率
    count,correct,handwritten,digital,printed = 0,0,0,0,0
    for j, l1 in enumerate(orderedLine1):
        predict = l1[1]
        label = orderedLine2[j][1]
        if label != 'none':
            count += 1
            if predict == label:
                correct += 1
                if predict == 'handwritten':
                    handwritten += 1
                elif predict == 'digital':
                    digital += 1
                elif predict == 'printed':
                    printed += 1
    print('共比较{}条非空记录, {}个预测正确, 准确率为{:.2%}'.format(count, correct, correct/count))
    print('正确预测分布: handwritten text - {}, handwritten digit - {}, printed text - {}'.format(handwritten, digital,printed))
    print('A total of {} valid records were evaluated, {} predictions were correct, the accuracy is {:2%}'.format(count, correct, correct/count))
    print('Correctly predict distribution: handwritten text - {}, handwritten digit - {}, printed text - {}'.format(handwritten,digital,printed))

if __name__ == '__main__':
    comparePreclassify(FilePaths.preclassifyFile, FilePaths.labelledFile)
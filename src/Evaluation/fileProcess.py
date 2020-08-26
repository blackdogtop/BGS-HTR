import jiwer

"""WER evaluation"""
referenceFile = '../../data/lines-v2.txt'  # reference
inputFile1 = '../../data/wordHTR-prediction.txt'  # word HTR
inputFile2 = '../../data/lineHTR-prediction.txt'  # line HTR
inputFile3 = '../../data/wordHTR-prediction-beamsearch.txt'  # beam search
inputFile4 = '../../data/wordHTR-prediction-wordbeamsearch.txt'  # word beam search
inputFile5 = '../../data/wordHTR-prediction-preprocess.txt'
inputFile6 = '../../data/wordHTR-prediction-beamsearch-preprocess.txt'
inputFile7 = '../../data/wordHTR-prediction-wordbeamsearch-preprocess.txt'
inputFile8 = '../../data/lineHTR-prediction-preprocess.txt'


def getLines(file):
    #  get lines
    with open(file, 'r') as f:
        lines = f.readlines()
    return lines


def compareImgNames(f1, f2):
    """compare image name is corresponded
    f1 is prediction file
    f2 is reference file
    """
    lines1 = getLines(f1)
    lines2 = getLines(f2)
    names1 = []
    names2 = []
    for l1 in lines1:
        if l1[0] == '#':
            continue
        if len(l1.split(' ')) <= 1:
            continue
        names1.append(l1.split()[0])
    for l2 in lines2:
        if l2[0] == '#':
            continue
        if l2.split(' ')[2] != 'handwritten':
            continue
        names2.append(l2.split()[0])
    # print(names1)
    # print(names2)
    if len(names1) != len(names2):
        print('File1 has {} lines, and File2 has {} lines'.format(len(names1), len(names2)))
        return 'length error'
    for i in range(len(names1)):
        if names1[i] != names2[i]:
            print(names1[i], names2[i])
            return 'name not corresponded'
    return 'correct'


def getValues(f1, f2):
    """get predictions and ground truth labels
    f1 is prediction file
    f2 is reference file
    """
    lines1 = getLines(f1)
    lines2 = getLines(f2)
    values1 = []
    values2 = []
    for l1 in lines1:
        if l1[0] == '#':
            continue
        if len(l1.split(' ')) <= 1:
            continue
        try:
            prediction = l1.rstrip().split()[1].split('|')
            prediction = [' '.join(prediction)]
        except IndexError:
            prediction = [' ']
        values1.append(prediction)
    for l2 in lines2:
        if l2[0] == '#':
            continue
        if l2.split(' ')[2] != 'handwritten':
            continue
        reference = l2.rstrip().split()[-1].split('|')
        reference = [' '.join(reference)]
        values2.append(reference)
    return values1, values2


def rmPunctuation(values):
    """preprocess the list of words to RemovePunctuation"""
    newValues = []
    for v in values:
        newValue = jiwer.RemovePunctuation()(v)
        newValue = jiwer.Strip()(newValue)
        newValue = jiwer.RemoveMultipleSpaces()(newValue)
        newValues.append(newValue)
    return newValues


def toLowerCase(values):
    """convert to lowercase"""
    newValues = []
    for v in values:
        newValue = jiwer.ToLowerCase()(v)
        newValues.append(newValue)
    return newValues


def createTemporaryFile(values, fileName):
    """将文本行写入临时文件"""
    temName = str(fileName) + '.txt'
    with open(temName, 'w') as f:
        for i, v in enumerate(values):
            f.write(v[0])
            if i != len(values) - 1:
                f.write('\n')


def main(f1, f2):
    """
    f1 is prediction file
    f2 is reference file
    """
    r = compareImgNames(f1, f2)
    print(r)
    if r == 'correct':  # 在对比没有错误的情况进行
        prediction, reference = getValues(f1, f2)
        #  移除标点
        prediction = rmPunctuation(prediction)
        reference = rmPunctuation(reference)
        #  转为小写
        prediction = toLowerCase(prediction)
        reference = toLowerCase(reference)
        # 创建临时文件
        createTemporaryFile(prediction, 'prediction')
        createTemporaryFile(reference, 'reference')


if __name__ == '__main__':
    main(inputFile8, referenceFile)

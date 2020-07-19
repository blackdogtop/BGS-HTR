from __future__ import division
from __future__ import print_function

import json
import random
import os

import cv2
import numpy as np

from helpers import preprocessor


class FilePaths:
    """ Filenames and paths to data """
    fnCharList = '../model/charList.txt'
    fnWordCharList = '../model/wordCharList.txt'
    fnCorpus = '../data/corpus.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '/path/to/infer/data/'  # 测试图片


class Sample:
    """ Sample from the dataset """

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    """ Batch containing images and ground truth texts """

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:
    """ Loads data from data folder """

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        """ Loader for dataset at given location, preprocess images and text according to parameters """

        assert filePath[-1] == '/'

        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        chars = set()
        bad_samples = []
        # Read json lables file
        # Dataset folder should contain a labels.json file inside, with key is the file name of images and value is the label
        with open(filePath + 'test.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line or line[0] == '#':
                    continue

                lineSplit = line.strip().split(' ')
                assert len(lineSplit) >= 9

                # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
                fileNameSplit = lineSplit[0].split('-')
                fileName = filePath + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

                # GT text are columns starting at 9
                gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
                chars = chars.union(set(list(gtText)))

                # check if image is not empty
                if not os.path.getsize(fileName):
                    bad_samples.append(lineSplit[0] + '.png')
                    continue

                # put sample into list
                self.samples.append(Sample(gtText, fileName))

        self.charList = list(open(FilePaths.fnCharList).read())

        # Split into training and validation set: 90% - 10%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        print("Train on", len(self.trainSamples), "images. Validate on",
              len(self.validationSamples), "images.")

        # Number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 25000

        # Start with train set
        self.trainSet()

        # List of all chars in dataset
        #self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def trainSet(self):
        """ Switch to randomly chosen subset of training set """
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    def validationSet(self):
        """ Switch to validation set """
        self.currIdx = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        """ Current batch index and overall number of batches """
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        """ Iterator """
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        """ Iterator """
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [preprocessor(self.samples[i].filePath,
                             self.imgSize, binary=True) for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)

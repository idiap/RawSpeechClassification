#!/usr/bin/python3

## Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
## Written by S. Pavankumar Dubagunta <pavankumar [dot] dubagunta [at] idiap [dot] ch>
## and Mathew Magimai Doss <mathew [at] idiap [dot] ch>
## 
## This file is part of RawSpeechClassification.
## 
## RawSpeechClassification is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License version 3 as
## published by the Free Software Foundation.
## 
## RawSpeechClassification is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with RawSpeechClassification. If not, see <http://www.gnu.org/licenses/>.


import pickle
import numpy
import h5py
import os

class rawGenerator:
    def __init__ (self, featDir, batchSize=256, spliceSize=25, mode='train'):
        self.featDir = featDir
        self.batchSize = batchSize
        self.spliceSize = spliceSize
        self.mode = mode

        self.stdFloor = 1e-3
        self.context = (self.spliceSize-1)//2
        infoFile = self.featDir + '/info.npy'
        self.info = numpy.load(infoFile).item()
       
        ## Set attributes from info
        self.numUtterances = self.info['numUtterances']
        self.numFeats = self.info['numFeats']
        self.numLabels = self.info['numLabels']
        self.numSplit = self.info['numSplit']
        self.inputFeatDim = self.info['inputFeatDim']*self.spliceSize
        self.outputFeatDim = self.info['outputFeatDim']

        ## Compute number of steps
        self.numSteps = -(-self.numFeats//self.batchSize)

        numpy.random.seed(512)
        self.splitDataCounter = 0
   
        self.x = numpy.empty ((0, self.inputFeatDim), dtype=numpy.float32)
        self.y = numpy.empty (0, dtype=numpy.int32)
        self.batchPointer = 0
        self.doUpdateSplit = True

    def addContextNorm (self, feat):
        ## Add context to get the window size
        N = len(feat)

        ## Repeat feat[0], feat[-1] so that we get the same number of spliced feats
        feat = numpy.concatenate([numpy.tile(feat[0], (self.context,1)), \
                feat, numpy.tile(feat[-1], (self.context,1))])

        feat = numpy.lib.stride_tricks.as_strided(feat, strides=feat.strides, \
                shape=(N, self.inputFeatDim))

        std = feat.std(axis=-1)
        std[std<self.stdFloor] = self.stdFloor
        feat = ((feat.T - feat.mean(axis=-1))/std).T
        return feat

    ## Make the object iterable
    def __iter__ (self):
        return self

    ## Retrieve a mini batch
    def __next__ (self):
        if self.mode == 'train':
            while (self.batchPointer + self.batchSize >= len (self.x)):
                if not self.doUpdateSplit:
                    self.doUpdateSplit = True
                    break

                self.splitDataCounter += 1

                featFile = '{:s}/{:d}.x.h5'.format(self.featDir, self.splitDataCounter)
                labelFile = '{:s}/{:d}.y.h5'.format(self.featDir, self.splitDataCounter)

                with h5py.File(featFile,'r') as f:
                    featList = [self.addContextNorm(f[i].value) for i in f]
                x = numpy.vstack(featList)

                with h5py.File(labelFile,'r') as f:
                    labelList = [f[i].value for i in f]
                y = numpy.hstack(labelList)

                self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
                self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
                self.batchPointer = 0

                ## Shuffle data
                randomInd = numpy.array(range(len(self.x)))
                numpy.random.shuffle(randomInd)
                self.x = self.x[randomInd]
                self.y = self.y[randomInd]

                if self.splitDataCounter == self.numSplit:
                    self.splitDataCounter = 0
                    self.doUpdateSplit = False
            
            xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
            yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
            self.batchPointer += self.batchSize
            return (xMini, yMini)
       
        else: ## Test mode
            while True:
                if self.doUpdateSplit:
                    self.splitDataCounter += 1
                    uflFile = '{:s}/{:d}.pickle'.format(self.featDir, self.splitDataCounter)
                    self.f = open(uflFile,'rb')
                    self.doUpdateSplit = False
                try:
                    utt,feat,lab = pickle.load(self.f)
                    return (utt,self.addContextNorm(feat),lab)
                except EOFError:
                    self.f.close()
                    self.doUpdateSplit = True
                    if self.splitDataCounter == self.numSplit:
                        self.splitDataCounter = 0
                        raise StopIteration


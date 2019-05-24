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


import scipy.io.wavfile as wav
import pickle
import numpy
import wave
import h5py
import sys
import os

class wav2feat:
    def __init__ (self, wavLabListFile, featDir=None, param=None, mode='train'):
        self.wavLabListFile = wavLabListFile
        self.featDir = featDir
        self.mode = mode
        self.maxSplitDataSize = 100 ## Utterances

        if param == None:
            param = {'windowLength' : 10,     ## milliseconds (We do splicing later)
                     'windowShift'  : 10,     ## milliseconds. Keep this same as above.
                     'fs'           : 16000,  ## Sampling rate in Hertz
                     'stdFloor'     : 1e-3 }  ## Floor on standard deviation
            param['windowLengthSamples'] = int(param['windowLength'] * param['fs'] / 1000.0)
            param['windowShiftSamples'] = int(param['windowShift'] * param['fs'] / 1000.0)

        self.param = param

        self.wll = open(self.wavLabListFile)
        self.numFeats, self.numUtterances, self.numLabels = self.checkList(self.wavLabListFile)

        self.inputFeatDim = self.param['windowLengthSamples']
        self.outputFeatDim=1 if self.numLabels==2 else self.numLabels 

    ## Exit
    def __exit__ (self):
        self.wll.close()

    ## Feature extraction routine
    def extract (self, wavepath):
        ## Read data and labels
        fs, data = wav.read(wavepath)
    
        ## Append zeros to data if necessary (we add dither later)
        if len(data) < self.param['windowLengthSamples']:
            data = numpy.concatenate([data, numpy.zeros(self.param['windowLengthSamples']-len(data))])
    
        ## Determine the number of frames, each of windowshift length
        numFeats = (len(data)-self.param['windowLengthSamples'])//self.param['windowShiftSamples']+1

        ## Convert Channel-1 of data into a feature matrix
        stride = data.strides[-1]
        feat = numpy.lib.stride_tricks.as_strided(data, shape=(numFeats, self.param['windowLengthSamples']), \
                strides=(self.param['windowShiftSamples']*stride,stride))
        feat = feat.astype(numpy.float32)
    
        ## Add dither
        feat += numpy.random.randn(numFeats, self.param['windowLengthSamples'])
    
        ## Mean normalise feature matrix
        feat = (feat.T - feat.mean(axis=-1)).T

        return feat

    ## Check files in list and return attributes
    def checkList (self, wavLabListFile):
        print ('Checking files in {:s}'.format(wavLabListFile))
        labels = set()
        numFeats = 0
        numUtterances = 0
        for wl in self.wll:
            w,l = wl.split()
            
            with wave.open(w) as f:
                ## Check number of channels and sampling rate
                assert f.getnchannels()==1, 'ERROR: {:s} has multiple channels. Modify the code accordingly and re-run'.format(w)
                assert f.getframerate()==self.param['fs'], 'ERROR: Sampling frequency mismatch with {:s}: '\
                        'expected {:f}, got {:f}'.format(w, self.param['fs'], f.getframerate())
                N = f.getnframes()
                
            numFeats += max((N-self.param['windowLengthSamples'])//self.param['windowShiftSamples']+1, 1)
            numUtterances += 1
            labels.update(l)
        numLabels = len(labels)
        self.wll.seek(0)
        return numFeats, numUtterances, numLabels

    ## Prepare feature directory for training/testing
    def prepareFeatDir (self):
        ## Create output directory
        os.makedirs(self.featDir, exist_ok=False)
        self.numSplit = -(-self.numUtterances//self.maxSplitDataSize)

        ## Save info
        self.info = {'numFeats': self.numFeats, 'numUtterances': self.numUtterances, 'numLabels':self.numLabels,\
                        'numSplit':self.numSplit, 'inputFeatDim':self.inputFeatDim, 'outputFeatDim':self.outputFeatDim}
        print(self.info)
        infoFile = '{:s}/info.npy'.format(self.featDir)
        numpy.save(infoFile, self.info)
       
        self.wll.seek(0) ## In case the object is used as iterator before calling this routine
        for self.splitDataCounter in range(1,self.numSplit+1):
            self.saveNextSplitData()
        self.wll.seek(0) ## For future use

    ## Process (return) feature and label for one utterance
    def processUtterance (self, wl):
        if not wl:
            return None,None
        w,l = wl.split()
        feat = self.extract(w)
        return w,feat,int(l)*numpy.ones(len(feat), dtype=numpy.int32)

    ## Save a split
    def saveNextSplitData (self):
        lines = [self.wll.readline() for n in range(self.maxSplitDataSize)]
        featLabList = [self.processUtterance(wl) for wl in lines if wl]

        if self.mode == 'train':
            uttList,featList,labelList = map(list,zip(*featLabList))
            featFile = '{:s}/{:d}.x.h5'.format(self.featDir, self.splitDataCounter)
            labelFile = '{:s}/{:d}.y.h5'.format(self.featDir, self.splitDataCounter)
            
            ## Save features
            with h5py.File(featFile, 'w') as f:
                for i,feat in enumerate(featList):
                    f.create_dataset(str(i), data=feat, dtype='float32')

            ## Save labels
            with h5py.File(labelFile, 'w') as f:
                for i,labels in enumerate(labelList):
                    f.create_dataset(str(i), data=labels, dtype='int32')

        else:
            featFile = '{:s}/{:d}.pickle'.format(self.featDir, self.splitDataCounter)
            with open(featFile,'wb') as f:
                for ufl in featLabList:
                    pickle.dump(ufl, f)

    ## Make the object iterable and retrieve one utterance each time
    def __iter__ (self):
        for wl in self.wll:
            yield processUtterance(wl)

## Main function prepares feature directory
if __name__ == '__main__':
    wavLabListFile = sys.argv[1]
    featDir = sys.argv[2]
    mode = sys.argv[3]

    w2f = wav2feat(wavLabListFile, featDir=featDir, mode=mode)
    w2f.prepareFeatDir()


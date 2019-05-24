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


import sys
import keras
import numpy
from rawGenerator import rawGenerator

def test (test_dir, model):
    r = rawGenerator(test_dir, mode='test')
    m = keras.models.load_model (model)

    spk_scores = {}
    spk_labels = {}
    spk_counts = {}
    for w,feat,l in r:
        pred = m.predict(feat)
        spk = w ## Get the speaker ID. This is useful when each speaker has multiple utterances and
                ## the results need to be calculated per speaker instead of per utterance. You need
                ## to configure this line according how the speaker ID can be extracted from you data.
                ## For e.g. the below line assumes that the basenames of the files start with speaker ID
                ## followed by an utterance ID, separated by a '_'.
                ## spk = w.split('/')[-1].split('_')[0]
                ## By default, we use the wav file name as the speaker ID, which means that
                ## each wav file corresponds to one speaker.
        if spk not in spk_scores:
            spk_scores[spk] = numpy.sum(pred, axis=0)
            spk_counts[spk] = len(pred)
            ## NOTE: Assuming the utterance labels are same across each speaker.
            ## Takes the label of the speaker's first utterance encountered.
            spk_labels[spk] = l[0]
        else:
            spk_scores[spk] += numpy.sum(pred, axis=0)
            spk_counts[spk] += len(pred)

    for spk in spk_labels:
        print (spk, spk_labels[spk], spk_scores[spk]/spk_counts[spk])
        
if __name__ == '__main__':
    test_dir = sys.argv[1]
    model = sys.argv[2]
    
    test (test_dir, model)

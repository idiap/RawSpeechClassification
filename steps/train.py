#!/usr/bin/env python3
# coding=utf-8

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


import inspect
import os
import sys

import keras
import keras.backend as K
import numpy

from keras.optimizers import SGD

from model_architecture import model_architecture
from rawGenerator import rawGenerator


if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 5:
    raise TypeError ('USAGE: train.py tr_dir cv_dir dnn_dir arch')

tr_dir  = sys.argv[1]
cv_dir  = sys.argv[2]
exp     = sys.argv[3]
arch    = sys.argv[4]

## Learning parameters
learning = {'rate'        : 0.1,    ## Initial learning rate
            'minEpoch'    : 5,      ## Minimum epochs to run before reducing learning rate
            'lrScale'     : 0.5,    ## Scale factor to learning rate
            'batchSize'   : 256,    ## Batch size
            'minValError' : 0.002,  ## Threshold on validation loss reduction between
                                    ## successive epochs, below which learning rate is scaled.
            'minLr'       : 1e-7}   ## The final learning rate below which the training stops.
## Number of times the learning rate has to be scaled.
learning['lrScaleCount'] = int(numpy.ceil(numpy.log(learning['minLr']/learning['rate']) / \
                            numpy.log(learning['lrScale'])))

os.makedirs (exp, exist_ok=True)
logger = keras.callbacks.CSVLogger(exp + "/log.dat", separator=" ", append=True)

cvGen = rawGenerator (cv_dir, learning['batchSize'])
trGen = rawGenerator (tr_dir, learning['batchSize'])

## Initialise learning parameters and models
sgd_args = inspect.getfullargspec(SGD)

if "learning_rate" in sgd_args.args:
    s = SGD(learning_rate=learning['rate'], weight_decay=0, momentum=0.5, nesterov=False)
else:
    s = SGD(lr=learning['rate'], decay=0, momentum=0.5, nesterov=False)

## Initialise model
numpy.random.seed(512)
m = model_architecture(arch, trGen.inputFeatDim, trGen.outputFeatDim)

## Initial training for "minEpoch-1" epochs
loss = 'binary_crossentropy' if trGen.outputFeatDim==1 else 'sparse_categorical_crossentropy'
m.compile(loss=loss, optimizer=s, metrics=['accuracy'])
print ('Learning rate: %f' % learning['rate'])
h = [
    m.fit_generator(
        trGen,
        steps_per_epoch=trGen.numSteps,
        validation_data=cvGen,
        validation_steps=cvGen.numSteps,
        epochs=learning['minEpoch']-1,
        verbose=2,
        callbacks=[logger],
    )
]
m.save (exp + '/cnn.keras', overwrite=True)
sys.stdout.flush()
sys.stderr.flush()

valErrorDiff = 1 + learning['minValError'] ## Initialise

## Continue training till validation loss stagnates
while learning['lrScaleCount']:
    print ('Learning rate: %f' % learning['rate'])
    h.append(
        m.fit_generator(
            trGen,
            steps_per_epoch=trGen.numSteps,
            validation_data=cvGen,
            validation_steps=cvGen.numSteps,
            epochs=1,
            verbose=2,
            callbacks=[logger],
        )
    )
    m.save(exp + '/cnn.keras', overwrite=True)
    sys.stdout.flush()
    sys.stderr.flush()

    ## Check validation error and reduce learning rate if required
    valErrorDiff = h[-2].history['val_loss'][-1] - h[-1].history['val_loss'][-1]
    if valErrorDiff < learning['minValError']:
        learning['rate'] *= learning['lrScale']
        learning['lrScaleCount'] -= 1
        K.set_value(m.optimizer.lr, learning['rate'])

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


import keras

def model_architecture (arch, inputFeatDim=4000, outputFeatDim=1):
    if arch == 'subseg':
        m = keras.models.Sequential([
                keras.layers.Reshape ((inputFeatDim, 1), input_shape=(inputFeatDim,)),
                keras.layers.Conv1D(filters=128, kernel_size=30, strides=10),
                keras.layers.Activation('relu'),
                keras.layers.pooling.MaxPooling1D(2),
                keras.layers.Conv1D(filters=256, kernel_size=10, strides=5),
                keras.layers.Activation('relu'),
                keras.layers.pooling.MaxPooling1D(3),
                keras.layers.Conv1D(filters=512, kernel_size=4, strides=2),
                keras.layers.Activation('relu'),
                keras.layers.Conv1D(filters=512, kernel_size=3, strides=1),
                keras.layers.Activation('relu'),
                # keras.layers.Flatten(),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(10),
                keras.layers.Activation('relu')])

    elif arch == 'seg':
        m = keras.models.Sequential([
                keras.layers.Reshape ((inputFeatDim, 1), input_shape=(inputFeatDim,)),
                keras.layers.Conv1D(filters=128, kernel_size=300, strides=100),
                keras.layers.Activation('relu'),
                keras.layers.pooling.MaxPooling1D(2),
                keras.layers.Conv1D(filters=256, kernel_size=5, strides=2),
                keras.layers.Activation('relu'),
                keras.layers.Conv1D(filters=512, kernel_size=4, strides=2),
                keras.layers.Activation('relu'),
                keras.layers.Conv1D(filters=512, kernel_size=3, strides=1),
                keras.layers.Activation('relu'),
                # keras.layers.Flatten(),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(10),
                keras.layers.Activation('relu')])

    else:
        raise TypeError('Unknown architecture: '+arch)

    ## Add the final layer
    m.add(keras.layers.Dense(outputFeatDim))
    if outputFeatDim == 1:
        m.add(keras.layers.Activation('sigmoid'))
    else:
        m.add(keras.layers.Activation('softmax'))


    return m

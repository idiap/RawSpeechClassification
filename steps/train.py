#!/usr/bin/env python3
# coding=utf-8

## Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
## Written by S. Pavankumar Dubagunta <pavankumar [dot] dubagunta [at] idiap [dot] ch>
## and Mathew Magimai Doss <mathew [at] idiap [dot] ch>
## and Olivier Bornet <olivier [dot] bornet [at] idiap [dot] ch>
## and Olivier Canévet <olivier [dot] canevet [at] idiap [dot] ch>
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

import argparse
import inspect
import os
import sys

from pathlib import Path

import keras
import keras.backend as K
import numpy as np

from keras.optimizers import SGD
from packaging.version import Version

from model_architecture import model_architecture

if Version(keras.__version__) < Version("3"):
    from rawGenerator import rawGenerator
else:
    from rawdataset import RawDataset as rawGenerator


parser = argparse.ArgumentParser(description="Train and validate the model")
parser.add_argument(
    "--train-feature-dir", required=True,
    help="Path to the directory containing the features for training"
)
parser.add_argument(
    "--validation-feature-dir", required=True,
    help="Path to the directory containing the features for validation"
)
parser.add_argument(
    "--arch", default="seg",
    help="Model architecture name"
)
parser.add_argument(
    "--slice-split", type=int, default=25,
    help="Slice size for feature context"
)
parser.add_argument(
    "--learning-rate", type=float, default=0.1,
    help="Initial learning rate"
)
parser.add_argument(
    "--learning-scale", type=float, default=0.5,
    help="Factor by which to reduce the learning rate"
)
parser.add_argument(
    "--batch-size", type=int, default=256,
    help="Batch size"
)
parser.add_argument(
    "--min-epoch", type=int, default=5,
    help="Minimum epochs to run before reducing learning rate"
)
parser.add_argument(
    "--output-dir", default="output-results",
    help="Output directory"
)
parser.add_argument(
    "--verbose", type=int, default=0,
    help="Keras verbose level for fit and predict"
)
args = parser.parse_args()


if __name__ != "__main__":
    raise ImportError("This script can only be run, and can't be imported")


tr_dir = args.train_feature_dir
cv_dir = args.validation_feature_dir
exp = args.output_dir
arch = args.arch
verbose = args.verbose

model_filename = Path(exp) / "cnn.keras"

## Learning parameters
learning = {
    "rate": args.learning_rate,
    "minEpoch": args.min_epoch,
    "lrScale": args.learning_scale,
    "batchSize": args.batch_size,
    ## Threshold on validation loss reduction between
    ## successive epochs, below which learning rate is scaled.
    "minValError": 0.002,
    "minLr": 1e-7,
}  ## The final learning rate below which the training stops.

## Number of times the learning rate has to be scaled.
learning["lrScaleCount"] = int(
    np.ceil(np.log(learning["minLr"] / learning["rate"]) / np.log(learning["lrScale"]))
)

Path(exp).mkdir(exist_ok=True, parents=True)
logger = keras.callbacks.CSVLogger(Path(exp) / "log.dat", separator=" ", append=True)

cvGen = rawGenerator(cv_dir, learning["batchSize"])
trGen = rawGenerator(tr_dir, learning["batchSize"])

## Initialise learning parameters and models
sgd_args = inspect.getfullargspec(SGD)

if "learning_rate" in sgd_args.args:
    s = SGD(
        learning_rate=learning["rate"], weight_decay=0, momentum=0.5, nesterov=False
    )
else:
    s = SGD(lr=learning["rate"], decay=0, momentum=0.5, nesterov=False)

## Initialise model
np.random.seed(512)
m = model_architecture(arch, trGen.inputFeatDim, trGen.outputFeatDim)

## Initial training for "minEpoch-1" epochs
loss = (
    "binary_crossentropy"
    if trGen.outputFeatDim == 1
    else "sparse_categorical_crossentropy"
)
m.compile(loss=loss, optimizer=s, metrics=["accuracy"])
print("Learning rate: %f" % learning["rate"])

if Version(keras.__version__) < Version("2.5.0"):
    output = m.fit_generator(
        trGen,
        steps_per_epoch=trGen.numSteps,
        validation_data=cvGen,
        validation_steps=cvGen.numSteps,
        epochs=learning["minEpoch"] - 1,
        verbose=verbose,
        callbacks=[logger],
    )
elif Version(keras.__version__) < Version("3"):
    output = m.fit(
        trGen,
        steps_per_epoch=trGen.numSteps,
        validation_data=cvGen,
        validation_steps=cvGen.numSteps,
        epochs=learning["minEpoch"] - 1,
        verbose=verbose,
        callbacks=[logger],
    )
else:
    output = m.fit(
        trGen,
        validation_data=cvGen,
        epochs=learning["minEpoch"] - 1,
        verbose=verbose,
        shuffle=False,
        callbacks=[logger],
    )

h = [output]


m.save(model_filename, overwrite=True)
sys.stdout.flush()
sys.stderr.flush()

valErrorDiff = 1 + learning["minValError"]  ## Initialise

## Continue training till validation loss stagnates
while learning["lrScaleCount"]:
    print("Learning rate: %f" % learning["rate"])
    if Version(keras.__version__) < Version("2.5.0"):
        output = m.fit_generator(
            trGen,
            steps_per_epoch=trGen.numSteps,
            validation_data=cvGen,
            validation_steps=cvGen.numSteps,
            epochs=1,
            verbose=verbose,
            callbacks=[logger],
        )
    elif Version(keras.__version__) < Version("3"):
        output = m.fit(
            trGen,
            steps_per_epoch=trGen.numSteps,
            validation_data=cvGen,
            validation_steps=cvGen.numSteps,
            epochs=1,
            verbose=verbose,
            callbacks=[logger],
        )
    else:
        output = m.fit(
            trGen,
            validation_data=cvGen,
            epochs=1,
            verbose=verbose,
            shuffle=False,
            callbacks=[logger],
        )
    h.append(output)

    m.save(model_filename, overwrite=True)
    sys.stdout.flush()
    sys.stderr.flush()

    ## Check validation error and reduce learning rate if required
    valErrorDiff = h[-2].history["val_loss"][-1] - h[-1].history["val_loss"][-1]
    if valErrorDiff < learning["minValError"]:
        learning["rate"] *= learning["lrScale"]
        learning["lrScaleCount"] -= 1
        if Version(keras.__version__) < Version("3"):
            K.set_value(m.optimizer.lr, learning["rate"])
        else:
            m.optimizer.learning_rate = learning["rate"]

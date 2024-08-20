# SPDX-FileCopyrightText: Copyright © Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: S. Pavankumar Dubagunta <pavankumar.dubagunta@idiap.ch>
# SPDX-FileContributor: Mathew Magimai Doss <mathew@idiap.ch>
# SPDX-FileContributor: Olivier Bornet <olivier.bornet@idiap.ch>
# SPDX-FileContributor: Olivier Canévet <olivier.canevet@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import pickle

from pathlib import Path

import h5py
import keras
import numpy as np


class RawDataset(keras.utils.PyDataset):
    def __init__(self, featDir, batchSize=256, spliceSize=25, mode="train", **kwargs):
        kwargs = {"workers": 1}
        super().__init__(**kwargs)
        self.featDir = featDir
        self.batchSize = batchSize
        self.spliceSize = spliceSize
        self.mode = mode

        self.stdFloor = 1e-3
        self.context = (self.spliceSize - 1) // 2
        infoFile = Path(self.featDir) / "info.npy"
        self.info = np.load(infoFile, allow_pickle=True).item()

        # Set attributes from info
        self.numUtterances = self.info["numUtterances"]
        self.numFeats = self.info["numFeats"]
        self.numLabels = self.info["numLabels"]
        self.numSplit = self.info["numSplit"]
        self.inputFeatDim = self.info["inputFeatDim"] * self.spliceSize
        self.outputFeatDim = self.info["outputFeatDim"]

        # Compute number of steps
        self.numSteps = -(-self.numFeats // self.batchSize)
        self.numDone = 0

        np.random.seed(512)
        self.splitDataCounter = 0

        self.x = np.empty((0, self.inputFeatDim), dtype=np.float32)
        self.y = np.empty(0, dtype=np.int32)
        self.batchPointer = 0
        self.doUpdateSplit = True

    def addContextNorm(self, feat):
        # Add context to get the window size
        N = len(feat)

        # Repeat feat[0], feat[-1] so that we get the same number of spliced feats
        feat = np.concatenate(
            [
                np.tile(feat[0], (self.context, 1)),
                feat,
                np.tile(feat[-1], (self.context, 1)),
            ],
        )

        feat = np.lib.stride_tricks.as_strided(
            feat,
            strides=feat.strides,
            shape=(N, self.inputFeatDim),
        )

        std = feat.std(axis=-1)
        std[std < self.stdFloor] = self.stdFloor
        return ((feat.T - feat.mean(axis=-1)) / std).T

    def __len__(self):
        return self.numSteps

    # Retrieve a mini batch
    def __getitem__(self, idx):
        self.numDone += 1
        if self.mode == "train":
            while self.batchPointer + self.batchSize >= len(self.x):
                if not self.doUpdateSplit:
                    self.doUpdateSplit = True
                    break

                self.splitDataCounter += 1

                featFile = f"{self.featDir}/{self.splitDataCounter}.x.h5"
                labelFile = f"{self.featDir}/{self.splitDataCounter}.y.h5"

                with h5py.File(featFile, "r") as f:
                    featList = [self.addContextNorm(f[i][()]) for i in f]
                x = np.vstack(featList)

                with h5py.File(labelFile, "r") as f:
                    labelList = [f[i][()] for i in f]
                y = np.hstack(labelList)

                self.x = np.concatenate((self.x[self.batchPointer :], x))
                self.y = np.concatenate((self.y[self.batchPointer :], y))
                self.batchPointer = 0

                # Shuffle data
                randomInd = np.array(range(len(self.x)))
                np.random.shuffle(randomInd)
                self.x = self.x[randomInd]
                self.y = self.y[randomInd]

                if self.splitDataCounter == self.numSplit:
                    self.splitDataCounter = 0
                    self.doUpdateSplit = False

            xMini = self.x[self.batchPointer : self.batchPointer + self.batchSize]
            yMini = self.y[self.batchPointer : self.batchPointer + self.batchSize]
            self.batchPointer += self.batchSize
            return (xMini, yMini)

        else:  # Test mode # noqa: RET505
            while True:
                if self.doUpdateSplit:
                    self.splitDataCounter += 1
                    uflFile = f"{self.featDir}/{self.splitDataCounter}.pickle"
                    self.f = open(uflFile, "rb")
                    self.doUpdateSplit = False
                try:
                    utt, feat, lab = pickle.load(self.f)
                    return utt, self.addContextNorm(feat), lab
                except EOFError:
                    self.f.close()
                    self.doUpdateSplit = True
                    if self.splitDataCounter == self.numSplit:
                        self.splitDataCounter = 0
                        raise StopIteration

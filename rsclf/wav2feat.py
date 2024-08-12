#!/usr/bin/env python3
# coding=utf-8

# Copyright (c) 2018-2024 Idiap Research Institute <https://www.idiap.ch>
# Written by S. Pavankumar Dubagunta <pavankumar [dot] dubagunta [at] idiap [dot] ch>
# and Mathew Magimai Doss <mathew [at] idiap [dot] ch>
# and Olivier Canévet <olivier [dot] canevet [at] idiap [dot] ch>
#
# This file is part of RawSpeechClassification.
#
# RawSpeechClassification is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# RawSpeechClassification is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RawSpeechClassification. If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import pickle

import h5py
import numpy
import scipy.io.wavfile as wav
import wave


class WAV2featExtractor:
    def __init__(self, wavLabListFile, featDir=None, param=None, mode="train"):
        self.wavLabListFile = wavLabListFile
        self.featDir = featDir
        self.mode = mode
        self.maxSplitDataSize = 100  # Utterances

        if param is None:
            param = {
                "windowLength": 10,  # milliseconds (We do splicing later)
                "windowShift": 10,  # milliseconds. Keep this same as above.
                "fs": 16000,  # Sampling rate in Hertz
                "stdFloor": 1e-3,
            }  # Floor on standard deviation
            param["windowLengthSamples"] = int(
                param["windowLength"] * param["fs"] / 1000.0
            )
            param["windowShiftSamples"] = int(
                param["windowShift"] * param["fs"] / 1000.0
            )

        self.param = param

        self.wll = open(self.wavLabListFile)
        self.numFeats, self.numUtterances, self.numLabels = self.checkList(
            self.wavLabListFile
        )

        self.inputFeatDim = self.param["windowLengthSamples"]
        self.outputFeatDim = 1 if self.numLabels == 2 else self.numLabels

    # Exit
    def __exit__(self):
        self.wll.close()

    # Feature extraction routine
    def extract(self, wavepath):
        # Read data and labels
        fs, data = wav.read(wavepath)

        # Append zeros to data if necessary (we add dither later)
        if len(data) < self.param["windowLengthSamples"]:
            data = numpy.concatenate(
                [data, numpy.zeros(self.param["windowLengthSamples"] - len(data))]
            )

        # Determine the number of frames, each of windowshift length
        numFeats = (len(data) - self.param["windowLengthSamples"]) // self.param[
            "windowShiftSamples"
        ] + 1

        # Convert Channel-1 of data into a feature matrix
        stride = data.strides[-1]
        feat = numpy.lib.stride_tricks.as_strided(
            data,
            shape=(numFeats, self.param["windowLengthSamples"]),
            strides=(self.param["windowShiftSamples"] * stride, stride),
        )
        feat = feat.astype(numpy.float32)

        # Add dither
        feat += numpy.random.randn(numFeats, self.param["windowLengthSamples"])

        # Mean normalise feature matrix
        feat = (feat.T - feat.mean(axis=-1)).T

        return feat

    # Check files in list and return attributes
    def checkList(self, wavLabListFile):
        print(f"Checking files in {wavLabListFile}")
        labels = set()
        numFeats = 0
        numUtterances = 0
        for wl in self.wll:
            w, label = wl.split()

            with wave.open(w) as f:
                # Check number of channels and sampling rate
                msg = (
                    f"ERROR: {w} has multiple channels. "
                    f"Modify the code accordingly and re-run"
                )
                assert f.getnchannels() == 1, msg
                msg = (
                    f"ERROR: Sampling frequency mismatch with {w}: "
                    f"expected {self.param['fs']}, got {f.getframerate()}"
                )
                assert f.getframerate() == self.param["fs"], msg
                N = f.getnframes()

            numFeats += max(
                (N - self.param["windowLengthSamples"])
                // self.param["windowShiftSamples"]
                + 1,
                1,
            )
            numUtterances += 1
            labels.update(label)
        numLabels = len(labels)
        self.wll.seek(0)
        return numFeats, numUtterances, numLabels

    # Prepare feature directory for training/testing
    def prepareFeatDir(self):
        # Create output directory
        os.makedirs(self.featDir, exist_ok=False)
        self.numSplit = -(-self.numUtterances // self.maxSplitDataSize)

        # Save info
        self.info = {
            "numFeats": self.numFeats,
            "numUtterances": self.numUtterances,
            "numLabels": self.numLabels,
            "numSplit": self.numSplit,
            "inputFeatDim": self.inputFeatDim,
            "outputFeatDim": self.outputFeatDim,
        }
        print(self.info)
        infoFile = f"{self.featDir}/info.npy"
        numpy.save(infoFile, self.info)

        # In case the object is used as iterator before calling this routine
        self.wll.seek(0)
        for self.splitDataCounter in range(1, self.numSplit + 1):
            self.saveNextSplitData()
        self.wll.seek(0)  # For future use

    # Process (return) feature and label for one utterance
    def processUtterance(self, wl):
        if not wl:
            return None, None
        w, label = wl.split()
        feat = self.extract(w)
        return w, feat, int(label) * numpy.ones(len(feat), dtype=numpy.int32)

    # Save a split
    def saveNextSplitData(self):
        lines = [self.wll.readline() for n in range(self.maxSplitDataSize)]
        featLabList = [self.processUtterance(wl) for wl in lines if wl]

        if self.mode == "train":
            uttList, featList, labelList = map(list, zip(*featLabList))
            featFile = f"{self.featDir}/{self.splitDataCounter}.x.h5"
            labelFile = f"{self.featDir}/{self.splitDataCounter}.y.h5"

            # Save features
            with h5py.File(featFile, "w") as f:
                for i, feat in enumerate(featList):
                    f.create_dataset(str(i), data=feat, dtype="float32")

            # Save labels
            with h5py.File(labelFile, "w") as f:
                for i, labels in enumerate(labelList):
                    f.create_dataset(str(i), data=labels, dtype="int32")

        else:
            featFile = f"{self.featDir}/{self.splitDataCounter}.pickle"
            with open(featFile, "wb") as f:
                for ufl in featLabList:
                    pickle.dump(ufl, f)

    # Make the object iterable and retrieve one utterance each time
    def __iter__(self):
        for wl in self.wll:
            yield self.processUtterance(wl)


def main():
    parser = argparse.ArgumentParser(description="Prepare the features")
    # fmt: off
    parser.add_argument(
        "--wav-list-file", required=True,
        help="Path to file containing on each row '/path/to/file.wav <label>'"
    )
    parser.add_argument(
        "--mode", required=True, choices=["train", "test"],
        help="Type of data"
    )
    parser.add_argument(
        "--feature-dir", default="output-features",
        help="Path where to save the features"
    )
    # fmt: on
    args = parser.parse_args()

    w2f = WAV2featExtractor(
        args.wav_list_file, featDir=args.feature_dir, mode=args.mode
    )
    w2f.prepareFeatDir()


if __name__ == "__main__":
    main()

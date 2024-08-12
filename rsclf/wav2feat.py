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

from pathlib import Path

import h5py
import numpy
import scipy.io.wavfile as wav
import wave


class WAV2featExtractor:
    """Extractor from wav files to features

    Args:

      wavLabListFile (str):
      featDir (str): Output directory where to save the features
      param:
      mode (str): Mode of the data 'train' or 'test'
      root (str): Prefix to append to the path

    """

    def __init__(self, wavLabListFile, featDir=None, param=None, mode="train", root=""):

        self.wavLabListFile = wavLabListFile
        self.featDir = featDir
        self.mode = mode
        self.maxSplitDataSize = 100  # Utterances
        self.root = root

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

    def __exit__(self):
        self.wll.close()

    def add_root(self, filename):
        if self.root:
            filename = str(Path(self.root) / filename)

        if not Path(filename).is_file():
            raise FileNotFoundError(f"File {filename} not found")

        return str(filename)

    def extract(self, wavepath):
        """Feature extraction routine"""

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

    def checkList(self, wavLabListFile):
        """Check files in list and return attributes"""

        print(f"Checking files in {wavLabListFile}")
        labels = set()
        numFeats = 0
        numUtterances = 0
        for wl in self.wll:
            w, label = wl.split()
            w = self.add_root(w)

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

    def prepareFeatDir(self):
        """Prepare feature directory for training/testing"""

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

    def processUtterance(self, wl):
        """Process (return) feature and label for one utterance"""

        if not wl:
            return None, None
        w, label = wl.split()
        w = self.add_root(w)
        feat = self.extract(w)
        return w, feat, int(label) * numpy.ones(len(feat), dtype=numpy.int32)

    def saveNextSplitData(self):
        """Save a split"""

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

    def __iter__(self):
        """Make the object iterable and retrieve one utterance each time"""
        for wl in self.wll:
            yield self.processUtterance(wl)


def main():
    parser = argparse.ArgumentParser(description="Prepare the features")
    # fmt: off
    parser.add_argument(
        "--root", default=None,
        help="Prefix to add in front of path names if provided. (e.g. root=/ssd/dataset/IEMOCAP)"
    )
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

    print(args)
    w2f = WAV2featExtractor(
        args.wav_list_file, featDir=args.feature_dir, mode=args.mode, root=args.root
    )
    w2f.prepareFeatDir()


if __name__ == "__main__":
    main()

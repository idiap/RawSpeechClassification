#!/usr/bin/env python3
# coding=utf-8

# Copyright (c) 2018-2024 Idiap Research Institute <https://www.idiap.ch>
# Written by S. Pavankumar Dubagunta <pavankumar [dot] dubagunta [at] idiap [dot] ch>
# and Mathew Magimai Doss <mathew [at] idiap [dot] ch>
# and Olivier Bornet <olivier [dot] bornet [at] idiap [dot] ch>
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

from pathlib import Path

import keras
import numpy as np

from rawdataset import RawDataset


def test(test_dir, model, output_dir, splice_size=25, verbose=0):
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    r = RawDataset(test_dir, splice_size=splice_size, mode="test")
    m = keras.models.load_model(model)

    spk_scores, spk_labels, spk_counts = {}, {}, {}
    for w, feat, l in r:
        pred = m.predict(feat, verbose=verbose)

        # Get the speaker ID. This is useful when each speaker has multiple utterances and
        # the results need to be calculated per speaker instead of per utterance. You need
        # to configure this line according how the speaker ID can be extracted from you data.
        # For e.g. the below line assumes that the basenames of the files start with speaker ID
        # followed by an utterance ID, separated by a '_'.
        # spk = w.split('/')[-1].split('_')[0]
        # By default, we use the wav file name as the speaker ID, which means that
        # each wav file corresponds to one speaker.
        spk = w

        if spk not in spk_scores:
            spk_scores[spk] = np.sum(pred, axis=0)
            spk_counts[spk] = len(pred)
            # NOTE: Assuming the utterance labels are same across each speaker.
            # Takes the label of the speaker's first utterance encountered.
            spk_labels[spk] = l[0]
        else:
            spk_scores[spk] += np.sum(pred, axis=0)
            spk_counts[spk] += len(pred)

    nb_correct = 0
    with open(Path(output_dir) / "scores.txt", "w") as f:
        for spk in spk_labels:
            label = spk_labels[spk]
            posterior = spk_scores[spk] / spk_counts[spk]
            prediction = np.argmax(posterior)
            print(spk, label, posterior, file=f)
            if prediction == label:
                nb_correct += 1

    accuracy = nb_correct / len(spk_labels)
    with open(Path(output_dir) / "accuracy.dat", "w") as f:
        print("accuracy", file=f)
        print(accuracy, file=f)


def main():
    parser = argparse.ArgumentParser(description="Test the model")
    # fmt: off
    parser.add_argument(
        "--feature-dir", required=True,
        help="Path to the directory containing the features"
    )
    parser.add_argument(
        "--model-filename", required=True,
        help="Path to the .keras model"
    )
    parser.add_argument(
        "--output-dir", default="output-results",
        help="Output directory"
    )
    parser.add_argument(
        "--splice-size", type=int, default=25,
        help="Slice size for feature context"
    )
    parser.add_argument(
        "--verbose", type=int, default=0,
        help="Keras verbose level for fit and predict"
    )
    # fmt: on
    args = parser.parse_args()

    test(args.feature_dir, args.model_filename, args.output_dir, args.splice_size, args.verbose)


if __name__ == "__main__":
    main()

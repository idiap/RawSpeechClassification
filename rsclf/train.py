#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright © Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: S. Pavankumar Dubagunta <pavankumar.dubagunta@idiap.ch>
# SPDX-FileContributor: Mathew Magimai Doss <mathew@idiap.ch>
# SPDX-FileContributor: Olivier Bornet <olivier.bornet@idiap.ch>
# SPDX-FileContributor: Olivier Canévet <olivier.canevet@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse

from pathlib import Path

import keras
import numpy as np

from keras.optimizers import SGD

from .model_architecture import model_architecture
from .rawdataset import RawDataset


def train(args):
    """Train the model by running `min-epoch` at a constant LR before reducing it."""
    tr_dir = args.train_feature_dir
    cv_dir = args.validation_feature_dir
    exp = args.output_dir
    arch = args.arch
    verbose = args.verbose

    model_filename = Path(exp) / "cnn.keras"

    # Learning parameters
    learning = {
        "rate": args.learning_rate,
        "minEpoch": args.min_epoch,
        "lrScale": args.learning_scale,
        "batchSize": args.batch_size,
        "spliceSize": args.splice_size,
        # Threshold on validation loss reduction between
        # successive epochs, below which learning rate is scaled.
        "minValError": 0.002,
        "minLr": 1e-7,
    }  # The final learning rate below which the training stops.

    # Number of times the learning rate has to be scaled.
    learning["lrScaleCount"] = int(
        np.ceil(
            np.log(learning["minLr"] / learning["rate"]) / np.log(learning["lrScale"]),
        ),
    )

    Path(exp).mkdir(exist_ok=True, parents=True)
    logger = keras.callbacks.CSVLogger(
        Path(exp) / "log.dat",
        separator=" ",
        append=True,
    )

    cvGen = RawDataset(cv_dir, learning["batchSize"], learning["spliceSize"])
    trGen = RawDataset(tr_dir, learning["batchSize"], learning["spliceSize"])

    s = SGD(
        learning_rate=learning["rate"],
        weight_decay=0,
        momentum=0.5,
        nesterov=False,
    )

    # Initialise model
    np.random.seed(512)
    m = model_architecture(arch, trGen.inputFeatDim, trGen.outputFeatDim)

    # Initial training for "minEpoch-1" epochs
    loss = (
        "binary_crossentropy"
        if trGen.outputFeatDim == 1
        else "sparse_categorical_crossentropy"
    )
    m.compile(loss=loss, optimizer=s, metrics=["accuracy"])
    print(f"Learning rate: {learning['rate']:g}")

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

    valErrorDiff = 1 + learning["minValError"]  # Initialise

    # Continue training till validation loss stagnates
    while learning["lrScaleCount"]:
        print(f"Learning rate: {learning['rate']:g}")
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

        # Check validation error and reduce learning rate if required
        valErrorDiff = h[-2].history["val_loss"][-1] - h[-1].history["val_loss"][-1]
        if valErrorDiff < learning["minValError"]:
            learning["rate"] *= learning["lrScale"]
            learning["lrScaleCount"] -= 1
            m.optimizer.learning_rate = learning["rate"]


def main():
    parser = argparse.ArgumentParser(description="Train and validate the model")
    # fmt: off
    parser.add_argument(
        "--train-feature-dir", required=True,
        help="Path to the directory containing the features for training",
    )
    parser.add_argument(
        "--validation-feature-dir", required=True,
        help="Path to the directory containing the features for validation",
    )
    parser.add_argument(
        "--arch", default="seg",
        help="Model architecture name",
    )
    parser.add_argument(
        "--splice-size", type=int, default=25,
        help="Slice size for feature context",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--learning-scale", type=float, default=0.5,
        help="Factor by which to reduce the learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--min-epoch", type=int, default=5,
        help="Minimum epochs to run before reducing learning rate",
    )
    parser.add_argument(
        "--output-dir", default="output-results",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose", type=int, default=0,
        help="Keras verbose level for fit and predict",
    )
    # fmt: on
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

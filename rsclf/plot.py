# coding=utf-8

# SPDX-FileCopyrightText: Copyright © Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Olivier Canévet <olivier.canevet@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


# Written by Olivier Canévet <olivier [dot] canevet [at] idiap [dot] ch>

import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def prepare_figure():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].set_ylabel("Training loss")
    axs[0, 0].grid(visible=True, linestyle="--")
    axs[0, 1].set_ylabel("Training accuracy")
    axs[0, 1].grid(visible=True, linestyle="--")
    axs[1, 0].set_ylabel("Validation loss")
    axs[1, 0].grid(visible=True, linestyle="--")
    axs[1, 1].set_ylabel("Validation accuracy")
    axs[1, 1].grid(visible=True, linestyle="--")
    return fig, axs

def main():
    """Create plots showing the performance of the classification for each epoch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".", help="Output directory for the plots file")
    parser.add_argument("--extension", default="png", help="Image extension to save")
    parser.add_argument("dirnames", nargs=argparse.REMAINDER, help="Location of the Keras trained model")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    fig, axs = prepare_figure()

    for dirname in args.dirnames:
        log_path = Path(dirname) / "log.dat"
        assert Path(log_path).is_file(), f"Expect file {log_path}"
        acc_path = Path(dirname) / "accuracy.dat"
        assert Path(acc_path).is_file(), f"Expect file {acc_path}"

        logs = pl.read_csv(log_path, has_header=True, separator=" ")
        axs[0, 0].plot(logs["loss"])
        axs[0, 1].plot(logs["accuracy"])
        axs[1, 0].plot(logs["val_loss"])
        axs[1, 1].plot(logs["val_accuracy"])

    fig.savefig(Path(args.output_dir) / f"plot.{args.extension}", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()

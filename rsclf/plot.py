# SPDX-FileCopyrightText: Copyright © Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Olivier Canévet <olivier.canevet@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Provide a command line interface for plotting results of an experiment."""

import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def prepare_figure():
    """Set up the axes, grid, and labels of the current figure."""
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
    """Create plots showing the performance of the classification for each epoch.

    Each given directory must contain a log.dat file and an accuracy.dat file.
    """
    parser = argparse.ArgumentParser(prog="rsclf-plot", description=main.__doc__)
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for the plots file",
    )
    parser.add_argument("--extension", default="png", help="Image extension to save")
    parser.add_argument(
        "dirnames",
        nargs="+",
        help="Location of the Keras trained model",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    fig, axs = prepare_figure()

    for dirname in args.dirnames:
        log_path = Path(dirname) / "log.dat"
        if not Path(log_path).is_file():
            raise FileNotFoundError(f"Expect file {log_path}")
        acc_path = Path(dirname) / "accuracy.dat"
        if not Path(acc_path).is_file():
            raise FileNotFoundError(f"Expect file {acc_path}")

        logs = pl.read_csv(log_path, has_header=True, separator=" ")
        axs[0, 0].plot(logs["loss"])
        axs[0, 1].plot(logs["accuracy"])
        axs[1, 0].plot(logs["val_loss"])
        axs[1, 1].plot(logs["val_accuracy"])

    fig.savefig(Path(args.output_dir) / f"plot.{args.extension}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

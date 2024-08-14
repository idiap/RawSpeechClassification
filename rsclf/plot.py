# coding=utf-8

# Copyright (c) 2024 Idiap Research Institute <https://www.idiap.ch>
# Written by Olivier Canévet <olivier [dot] canevet [at] idiap [dot] ch>
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

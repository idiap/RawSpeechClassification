<!--
SPDX-FileCopyrightText: Copyright © Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: S. Pavankumar Dubagunta <pavankumar.dubagunta@idiap.ch>
SPDX-FileContributor: Olivier Canévet <olivier.canevet@idiap.ch>
SPDX-FileContributor: Yannick Dayer <yannick.dayer@idiap.ch>

SPDX-License-Identifier: GPL-3.0-only
-->

# Raw Speech Classification

[![PyPI package](https://shields.io/pypi/v/raw-speech-classification.svg?logo=pypi)](https://pypi.org/project/raw-speech-classification)

Trains CNN (or any neural network based) classifiers from raw speech
using Keras and tests them. The inputs are lists of wav files, where
each file is labelled. It then creates fixed length signals and
processes them. During testing, it computes scores at the utterance or
speaker levels by averaging the corresponding frame-level scores from
the fixed length signals.

## Installation

### Installing from PyPI

If you want to install the last release of this package in your current environment (we
recommend using Conda to manage virtual environments), you can run either of the
following commands, depending on your desired framework:

```bash
pip install raw-speech-classification[torch]
```

or

```bash
pip install raw-speech-classification[tensorflow]
```

or

```bash
pip install raw-speech-classification[jax]
```

If you already have an environment with PyTorch, TensorFlow, or Jax
installed, you can simply run:

```bash
pip install raw-speech-classification
```

### Installing from source in a virtual environment

To install the following repository from source, clone this repository, then
run (according to the Keras backend you desire):

```bash
pip install -e .[torch]
```

or

```bash
pip install -e .[tensorflow]
```

or

```bash
pip install -e .[jax]
```

You will also need to set the `KERAS_BACKEND` environment variable to
the correct backend before running `rsclf-train` or `rsclf-test` (see
below), or globally for the current bash session with:

```bash
export KERAS_BACKEND=torch
```

Replace `torch` by `tensorflow` or `jax` accordingly.

## Using the code

1. Create lists for training, cross-validation and testing.
   Each line in a list must contain the path to a wav file (relative to the `-R` or
   `--root` option), followed by its integer label indexed from 0, separated by a space.
   E.g. if your data files are in `/home/bob/data/my_dataset/part*/file*.wav`, the
   `root` option could be `/home/bob/data/my_dataset` and the content of the files would
   then be like:

   ```text
   part1/file1.wav 1
   part1/file2.wav 0
   ```

   Full list files for IEMOCAP are available in the repository as example in
   [`datasets/IEMOCAP/F1_lists`](datasets/IEMOCAP/F1_lists).

1. You can now run the following commands (use the `--help` option of each command to get
   more details):

   ```bash
   rsclf-wav2feat --wav-list-file list_files/cv.list --feature-dir output/cv_feat --mode train --root path/to/dataset/basedir
   rsclf-wav2feat --wav-list-file list_files/train.list --feature-dir output/train_feat --mode train --root path/to/dataset/basedir
   rsclf-wav2feat --wav-list-file list_files/test.list --feature-dir output/test_feat --mode test --root path/to/dataset/basedir
   KERAS_BACKEND=torch rsclf-train --train-feature-dir output/train_feat --validation-feature-dir output/cv_feat --output-dir output/cnn_subseg --arch subseg --splice-size 25 --verbose 2
   KERAS_BACKEND=torch rsclf-test --feature-dir output/test_feat --model-filename output/cnn_subseg/cnn.keras --output-dir output/cnn_subseg --splice-size 25 --verbose 0
   rsclf-plot --output-dir output/ output/cnn_subseg
   ```

### Using `run.sh` (only with Conda)

`run.sh` allows to run all these steps in one command when using Conda as virtual
environment manager. This examples shows how to use the IEMOCAP lists shipped in the
repository:

```bash
bash run.sh -C ~/miniforge3 -n rscl -D datasets/IEMOCAP/F1_lists -a seg -o results/seg-f1 -R <IEMOCAP_DATA_ROOT>
```

Replace `<IEMOCAP_DATA_ROOT>` by the path to the IEMOCAP data directory containing
`IEMOCAP_full_release/Session*`.

[Here is an example](./docs/log.txt) of the log printed to the terminal, and you should
obtain the following curve in `results/seg-f1/plot.png`:

![Results](./docs/plot.png)

## Code components

1. [`wav2feat.py`](rsclf/wav2feat.py) creates directories where the wav files are stored
   as fixed length frames for faster access during training and testing.

1. [`train.py`](rsclf/train.py) is the Keras training script.

1. Model architecture can be configured in
   [`model_architecture.py`](rsclf/model_architecture.py).

1. [`rawdataset.py`](rsclf/rawdataset.py) provides an object that reads the saved
   directories in batches and retrieves mini-batches for training.

1. [`test.py`](rsclf/test.py) performs the testing and generates scores as posterior
   probabilities. If you need the results per speaker, configure it accordingly (see the
   script for details). The default output format is:

   ```text
    <speakerID> <label> [<posterior_probability_vector>]
   ```

1. [`plot.py`](rsclf/plot.py) generates and saves the learning curves.

## Training schedule

The script uses stochastic gradient descent with 0.5 momentum. It
starts with a learning rate of 0.1 for a minimum of 5 epochs. Whenever
the validation loss reduces by less than 0.002 between successive
epochs, the learning rate is halved. Halving is performed until the
learning rate reaches 1e-7.

## Contributors

Idiap Research Institute

Authors: S. Pavankumar Dubagunta and Dr. Mathew Magimai-Doss

## License

GNU GPL v3

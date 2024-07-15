# Raw Speech Classification

Trains CNN (or any neural network based) classifiers from raw speech
using Keras and tests them. The inputs are lists of wav files,
where each file is labelled. It then creates fixed length signals and
processes them. During testing, it computes scores at the
utterance or speaker levels by averaging the corresponding frame-level
scores from the fixed length signals.

## Installation

To install the legacy code for Tensorflow 1.12 and CUDA 9.0:

```bash
conda env create -f conda/legacy.yaml
```

## Dependencies

1. Python 3.4+.
2. Keras with Tensorflow/Theano backend.

## Using the Code

1. Create lists for training, cross-validation and testing.
   Each line in a list must contain the full path to a wav file,
   follwed by its integer label indexed from 0, separated by a space.
   E.g:

    \</path1/file1.wav\> 1

    \</path2/file2.wav\> 0

2. Configure and run run.sh. Provide model architecture
   as an argument. See steps\_kt/model\_architecture.py for valid
   options. Optionally, provide an integer as a count of the number of
   times the experiment is repeated. This is useful when the same
   experiment needs to be repeated multiple times with different
   initialisations. The argument defaults to 1.

## Code Components

1. wav2feat.py creates directories where the wav files are stored as
   fixed length frames for faster access during training and testing.

2. train.py is the Keras training script.

3. Model architecture can be configured in model\_architecture.py.

4. rawGenerator.py provides an object that reads the saved directories
   in batches and retrieves mini-batches for training.

5. test.py performs the testing and generates scores as posterior
   probabilities. If you need the results per speaker,
   configure it accordingly (see the script for details). The default
   output format is:

    \<speakerID\> \<label\> \[\<posterior\_probability\_vector\>\]

## Training Schedule

The script uses stochastic gradient descent with 0.5 momentum. It
starts with a learning rate of 0.1 for a minimum of 5
epochs. Whenever the validation loss reduces by less than 0.002
between successive epochs, the learning rate is halved. Halving
is performed until the learning rate reaches 1e-7.

## Contributors

Idiap Research Institute

Authors: S. Pavankumar Dubagunta and Dr. Mathew Magimai-Doss


## License

GNU GPL v3

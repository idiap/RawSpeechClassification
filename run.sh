#!/bin/bash

# The "main script" of RawSpeechClassification.

# Copyright (c) 2018 Idiap Research Institute <https://www.idiap.ch/>
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


export PYTHONPATH=$PWD/steps:$PYTHONPATH

arch="$1"       # Model architecture
iter="${2:-1}"

[ -z $arch ] && echo "Model architecture not provided. Defaulting to 'subseg'." && arch=subseg
exp=exp/cnn_${arch}_${iter}   # Output directory


# Wav file lists
train_list=lists/train.list
cv_list=lists/cv.list
test_list=lists/test.list

# Feature directories
train_feat=feat/train_feat
cv_feat=feat/cv_feat
test_feat=feat/test_feat

# Extract features
[ -d $cv_feat ] || ./steps/wav2feat.py $cv_list $cv_feat "train"
[ -d $train_feat ] || ./steps/wav2feat.py $train_list $train_feat "train"
[ -d $test_feat ] || ./steps/wav2feat.py $test_list $test_feat "test"

# Train
[ -f $exp/cnn.keras ] || ./steps/train.py $train_feat $cv_feat $exp $arch
[ ! -f $exp/cnn.keras ] && echo "Training failed. Check logs." && exit 1

# Test
[ -s $exp/scores.txt ] || ./steps/test.py $test_feat $exp/cnn.keras > $exp/scores.txt
[ ! -s $exp/scores.txt ] && echo "Testing failed. Check logs." && exit 1

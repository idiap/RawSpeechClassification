#!/bin/bash

# bash ${0} -C ~/miniconda3 -n rsclf

echo "Start to run on $(hostname) at $(date +'%F %T')"
SECONDS=0

CONDA_HOME=
CONDA_ENV=

arch="subseg"
iter=1
datadir=

######################################################################

while getopts ":C:n:o:a:D:" opt
do
  case ${opt} in
    C) CONDA_HOME="${OPTARG}" ;;
    n) CONDA_ENV="${OPTARG}" ;;
    o) OUTPUT="${OPTARG}" ;;
    a) arch="${OPTARG}" ;;
    D) datadir="${OPTARG}" ;;
    \?) echo "Invalid option -%s\n" "${OPTARG}" ; exit 1 ;;
  esac
done
shift $(( OPTIND - 1 ))

######################################################################

[[ -z ${CONDA_HOME} ]] && echo "Missing option -C (eg: ~/miniconda3)" && exit 1
[[ -z ${CONDA_ENV} ]] && echo "Missing option -n (eg: env_name)" && exit 1
[[ -z ${datadir} ]] && echo "Missing option -D (eg: dir containing train.list, test.list, cv.list)" && exit 1

######################################################################

eval "$(${CONDA_HOME}/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV}
export PYTHONPATH=$PWD/steps:$PYTHONPATH

if [[ `conda list | grep torch` ]] ; then
  echo "Using torch backend"
  export KERAS_BACKEND=torch
elif [[ `conda list | grep tensorflow` ]] ; then
  echo "Using tensorflow backend"
  export KERAS_BACKEND=tensorflow
fi

mkdir -p ${OUTPUT}

conda env export > ${OUTPUT}/env.yaml
conda list > ${OUTPUT}/env.txt

echo "Using architecture '${arch}'"

exp=${OUTPUT}/cnn_${arch}_${iter} # Output directory

# Wav file lists
train_list=${datadir}/train.list
cv_list=${datadir}/cv.list
test_list=${datadir}/test.list

# Feature directories
train_feat=${OUTPUT}/train_feat
cv_feat=${OUTPUT}/cv_feat
test_feat=${OUTPUT}/test_feat

spliceSize=25

# Extract features
[ -d $cv_feat ] || python3 steps/wav2feat.py --wav-list-file $cv_list --feature-dir $cv_feat --mode "train"
[ -d $train_feat ] || python3 steps/wav2feat.py --wav-list-file $train_list --feature-dir $train_feat --mode "train"
[ -d $test_feat ] || python3 steps/wav2feat.py --wav-list-file $test_list --feature-dir $test_feat --mode "test"

# Train
[ -f $exp/cnn.keras ] || python3 steps/train.py --train-feature-dir $train_feat --validation-feature-dir $cv_feat --output-dir $exp --arch $arch --splice-size $spliceSize --verbose 2
[ ! -f $exp/cnn.keras ] && echo "Training failed. Check logs." && exit 1

# Test
[ -s $exp/scores.txt ] || python3 steps/test.py --feature-dir $test_feat --model-filename $exp/cnn.keras --output-dir $exp --splice-size $spliceSize --verbose 0
[ ! -s $exp/scores.txt ] && echo "Testing failed. Check logs." && exit 1

echo "Script took $(date -u -d @${SECONDS} +"%T")"
echo "End at $(date +'%F %T')"

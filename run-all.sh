#!/bin/bash

# bash ${0} -C ~/miniconda3 -n rsclf

#SBATCH --time 08:00:00
#SBATCH --job-name raw-sp-clf
#SBATCH --mem 20G
#SBATCH --cpus-per-task 4
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --constraint "p40|v100"

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

# Extract features
[ -d $cv_feat ] || ./steps/wav2feat.py $cv_list $cv_feat "train"
[ -d $train_feat ] || ./steps/wav2feat.py $train_list $train_feat "train"
[ -d $test_feat ] || ./steps/wav2feat.py $test_list $test_feat "test"

# Train
[ -f $exp/cnn.h5 ] || ./steps/train.py $train_feat $cv_feat $exp $arch
[ ! -f $exp/cnn.h5 ] && echo "Training failed. Check logs." && exit 1

# Test
[ -s $exp/scores.txt ] || ./steps/test.py $test_feat $exp/cnn.h5 > $exp/scores.txt
[ ! -s $exp/scores.txt ] && echo "Testing failed. Check logs." && exit 1

echo "Script took $(date -u -d @${SECONDS} +"%T")"
echo "End at $(date +'%F %T')"

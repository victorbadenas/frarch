#!/bin/bash

usage="$(basename "$0") [-h] [-e exp_name] -- build experiment folder

where:
    -h  show this help text
    -e|--experiment experiment name"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -e|--experiment)
            EXPERIMENT_NAME="$2"
            [ -z $EXPERIMENT_NAME ] && echo "-e option requires non empty name" && echo "$usage" >&2 && exit 1;
            shift # past argument
            shift # past value
            ;;
        -h|--help)
            echo "$usage" >&2
            exit 0
            ;;
        *) printf "missing argument for $key\n" >&2
            echo "$usage" >&2
            exit 1
            ;;
    esac
done
# echo "EXPERIMENT_NAME = ${EXPERIMENT_NAME}"

# change to experiments folder
ROOT_FOLDER=$(realpath $(dirname "$0")/../)
cd $ROOT_FOLDER/experiments
pwd

if [ -d "$EXPERIMENT_NAME" ];
then
    echo -e "Experiment ${EXPERIMENT_NAME} exists"
    exit 1;
fi

mkdir -p ${EXPERIMENT_NAME}/hparams
mkdir -p ${EXPERIMENT_NAME}/results
cat $ROOT_FOLDER/scripts/train.py.template | sed "s/<experiment>/${EXPERIMENT_NAME}/g" | sed "s/<date>/$(date)/g" > $EXPERIMENT_NAME/train_$EXPERIMENT_NAME.py
cat $ROOT_FOLDER/scripts/hparams.yaml.template | sed "s/<experiment_name>/${EXPERIMENT_NAME}/g" | sed "s/<user_name>/$USER/g" > $EXPERIMENT_NAME/hparams/${EXPERIMENT_NAME}_train.yaml

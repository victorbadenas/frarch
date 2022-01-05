#!/usr/bin/env bash

set -e

test_training () {
    experimentFolder=$1
    cd ${experimentFolder}
    resultsFolder=$(cat train.yaml | grep -Po '^experiment_folder: "\K.*(?=")')
    python train.py train.yaml --device cpu --debug --debug_batches 100
    [ $(ls ${resultsFolder}/save/*/*.pt | wc -l) -gt 0 ]
    [ $(ls ${resultsFolder}/save/*/*.json | wc -l) -gt 0 ]
    [ $(ls ${resultsFolder}/train.yaml | wc -l) -gt 0 ]
    cd -
}

[ $(ls -d */results/ | wc -l) -gt 0 ] && rm -rf */results/
test_training train_mnist

set +e

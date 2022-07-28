#!/usr/bin/env bash

set -e

test_training () {
    experimentFolder=$1
    cd ${experimentFolder}
    resultsFolder=yq '.experiment_folder' train.yaml
    python train.py train.yaml --device cpu --debug --debug_batches 10
    [ $(ls ${resultsFolder}/save/*/*.pt | wc -l) -gt 0 ]
    [ $(ls ${resultsFolder}/save/*/*.json | wc -l) -gt 0 ]
    [ $(ls ${resultsFolder}/train.yaml | wc -l) -gt 0 ]
    cd -
}

[ $(ls -d */results/ | wc -l) -gt 0 ] && rm -rf */results/
test_training train_mnist

set +e

#!/usr/bin/env bash

set -e

exampleDir=$1

test_training () {
    experimentFolder=$1
    cd ${experimentFolder}
    python train.py hparams/train.yaml --device cpu --debug --debug_batches 10
    [ $(ls results/save/*/*.pt | wc -l) -gt 0 ]
    [ $(ls results/save/*/*.json | wc -l) -gt 0 ]
    [ $(ls results/train.yaml | wc -l) -gt 0 ]
    cd -
}

[ $(ls -d ${exampleDir}/results/ | wc -l) -gt 0 ] && rm -rf ${exampleDir}/results/
test_training $1

set +e

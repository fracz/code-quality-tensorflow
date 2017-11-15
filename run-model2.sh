#!/usr/bin/env bash

for D in input/*; do
    if [ -d "${D}" ]; then
        datasetName="${D##*/}"
        echo "*** DATASET: $datasetName ***"
        python model2.py "$datasetName"
    fi
done

#!/bin/bash

ARG_COMPILE=--${2:-'no-compile'}
ARG_BACKEND=--backend=${3:-'inductor'}

if [ "$1" = "vanilla_pretrain" ]; then
    echo "Starting Vanilla Pretrain: ~ 2 Hours"
    python run.py $ARG_COMPILE $ARG_BACKEND --function=pretrain --variant=vanilla --pretrain_corpus_path=./data/train_word.txt --writing_params_path=./submission/vanilla.pretrain.params
elif [ "$1" = "perceiver_pretrain" ]; then
    echo "Starting Perceiver Pretrain: ~ 2 Hours"
    python run.py $ARG_COMPILE $ARG_BACKEND --function=pretrain --variant=perceiver --pretrain_corpus_path=./data/train_word.txt --writing_params_path=./submission/perceiver.pretrain.params
else
    echo "Invalid Option Selected. Only Options Available Are:"
    echo "==============================================="
    echo "./run.sh vanilla_pretrain"
    echo "./run.sh perceiver_pretrain"
    echo "==============================================="
fi

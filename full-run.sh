#!/bin/env bash
make clean-lin
script -c "make scrape" /dev/null | tee scrape-log.txt
./train-run.sh hypfeatures --load-tokens=src/tokenizer.pickle --num-epochs=40 --context-filter="(no-args+(hyp-args%(etactic:apply+etactic:exploit+etactic:rewrite)))%default"

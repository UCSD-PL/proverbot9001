#!/bin/env bash
make clean-lin
script -c "make scrape" /dev/null | tee scrape-log.txt
./src/proverbot9001.py train hypfeatures data/scrape.txt data/hypfeatures-weights.dat --load-tokens=src/tokenizer.pickle --num-epochs=40 --context-filter="(no-args+(hyp-args%(etactic:apply+etactic:exploit+etactic:rewrite)))%default"
cat data/compcert-test-files.txt | xargs ./src/proverbot9001.py search-report --weightsfile=data/hypfeatures-weights.dat --prelude=./CompCert --search-width=3 --search-depth=6 -o search-report 
grep -o "Proofs Completed: *[0-9]\+\.[0-9]\+% ([0-9]\+/[0-9]\+)" search-report/report.html

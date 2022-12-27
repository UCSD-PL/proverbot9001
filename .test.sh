#!/bin/bash

date

git rev-parse HEAD

echo Hello from lambda.

# install rust

curl https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init > /tmp/rustup-init

chmod +x /tmp/rustup-init

/tmp/rustup-init -y

export PATH=$PATH:~/.cargo/bin:~/.local/bin

# need a virtualenv or else maturin won't run
# use site packages so we don't have to reinstall
# pytorch & friends
python3 -m venv --system-site-packages .


source bin/activate

# run setup
make setup

# okay, we've got an install.

make data/compcert-scrape.txt -j `nproc`

python3 ./src/proverbot9001.py tokens data/compcert-scrape.txt tokens.txt &
python3 ./src/proverbot9001.py tactics data/compcert-scrape.txt tactics.txt &

wait # generate tokens and tactics concurrently before proceeding to training

make compcert-train

python src/search_file.py --prelude ./CompCert/ lib/Parmov.v --weightsfile=data/polyarg-weights.dat --no-generate-report

echo proofs succeeded:
cat search-report/proofs.csv  | grep SUCCESS | wc -l

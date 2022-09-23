#!/bin/bash

# install rust

curl https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init > /tmp/rustup-init

chmod +x /tmp/rustup-init

/tmp/rustup-init -y

export PATH=$PATH:~/.cargo/bin

# run setup

make setup

# okay, we've got an install.

make data/compcert-scrape.txt -j `nproc`

python3 ./src/proverbot9001.py tokens data/compcert-scrape.txt tokens.txt
python3 ./src/proverbot9001.py tactics data/compcert-scrape.txt tactics.txt

make compcert-train

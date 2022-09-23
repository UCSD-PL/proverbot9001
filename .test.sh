#!/bin/bash

# install rust

curl https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init > /tmp/rustup-init

chmod +x /tmp/rustup-init

/tmp/rustup-init -y

export PATH=$PATH:~/.cargo/bin

# run setup

make setup

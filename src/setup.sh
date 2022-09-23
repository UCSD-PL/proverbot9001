#!/usr/bin/env bash
set -euv

echo "Making sure your environment is correctly setup"

if [[ -f /etc/NIXOS ]]; then
    if [[ -z ${NIXSHELL:=} ]]; then
        echo "Please run nix-shell to ensure your setup is correct"
        exit 1
    else
        continue
    fi
else
    git submodule init
    git submodule update
    opam init -a --compiler=4.07.1 -y
    eval `opam config env`
    opam update
    # For Coq:
    opam pin -yn add coq 8.10.2
    opam pin -yn add menhir 20190626
    # For SerAPI:
    opam install -y coq-serapi
    # Python dependencies
    pip3 install --no-input --user -r requirements.txt
    pip3 install --no-input -e coq_serapy
    # For py03/dataloader
    rustup toolchain install nightly
    make src/dataloader.so
fi

function check-and-clone {
    if [[ ! -d $1 ]]; then
        git clone $2
    fi
    (cd $1 && git fetch && git checkout $3) || exit 1
}
function setup-compcert {
    # check-and-clone\
    #     "CompCert" "https://github.com/AbsInt/CompCert.git"\
    #     "76a4ff8f5b37429a614a2a97f628d9d862c93f46"
    (
        set -euv
        cd CompCert
        if [[ ! -f "Makefile.config" ]]; then
            ./configure x86_64-linux
        fi
        make -j `nproc`
        ../src/patch_compcert.sh
    ) || exit 1
}

# setup-coq-menhir
setup-compcert

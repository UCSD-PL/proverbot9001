#!/usr/bin/env bash
set -euv

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

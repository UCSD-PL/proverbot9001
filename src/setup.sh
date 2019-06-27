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
    opam init -a --compiler=4.07.1
    eval `opam config env`
    # For Coq:
    opam pin add menhir 20181113
    opam install -y menhir
    # For SerAPI:
    opam install -y coq-serapi
    pip3 install --user -r requirements.txt
fi

function check-and-clone {
    if [[ ! -d $1 ]]; then
        git clone $2
    fi
    (cd $1 && git fetch && git checkout $3) || exit 1
}

function setup-coq-menhir {
    check-and-clone\
        "coq-menhirlib" "https://gitlab.inria.fr/fpottier/coq-menhirlib.git"\
        "ca0655b2f96057a271fb5c9a254a38d195b4a7f9"
    (
        set -euv
        cd coq-menhirlib
        make && make install
    )
}

function setup-compcert {
    check-and-clone\
        "CompCert" "https://github.com/AbsInt/CompCert.git"\
        "f047fcb7852ff58c0c62f10d41f91f3f88552780"
    (
        set -euv
        cd CompCert
        if [[ ! -f "Makefile.config" ]]; then
            PATH="$PWD/../coq/bin:$PATH" ./configure x86_64-linux
        fi
        PATH="$PWD/../coq/bin:$PATH" make -j `nproc`
    ) || exit 1
}

setup-coq-menhir
setup-compcert

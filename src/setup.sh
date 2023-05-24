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
    git submodule init coq_serapy
    git submodule init dataloader/gestalt-ratio
    git submodule init CompCert
    git submodule update
    opam init -a --compiler=4.07.1 -y
    eval `opam config env`
    opam update
    # For Coq:
    opam pin -yn add coq 8.10.2
    opam pin -yn add menhir 20190626
    # For SerAPI:
    opam install -y coq-serapi coq menhir
    # Python dependencies
    pip3 install --no-input --user -r requirements.txt
    pip3 install --no-input --user -e coq_serapy
    # For py03/dataloader
    rustup toolchain install nightly
    (cd dataloader/dataloader-core && maturin develop -r)
fi

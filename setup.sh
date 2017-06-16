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
  opam switch 4.04.0
  # For Coq:
  opam install menhir
  # For SerAPI:
  opam install ocamlfind ppx_deriving ppx_import cmdliner core_kernel sexplib ppx_sexp_conv camlp5
  eval `opam config env`
fi

function check-and-clone {
    if [[ ! -d $1 ]]; then
        git clone $2
    fi
    (cd $1 && git fetch && git checkout $3) || exit 1
}

function setup-coq {
  check-and-clone\
    "coq" "https://github.com/coq/coq.git"\
    "9d423562a5f83563198f3141500af4c97103c2bf"
  (
    set -euv
    cd coq
    if [ ! -f config/coq_config.ml ]; then
      ./configure -local
    fi
    make -j `nproc`
  ) || exit 1
}

function setup-coq-serapi {
  check-and-clone\
    "coq-serapi" "https://github.com/ejgallego/coq-serapi.git"\
    "3f4df09666d8f2cbd598f36872dbd0478efaf778"
  (
    set -euv
    cd coq-serapi
    echo "$PWD/../coq"
    SERAPI_COQ_HOME="$PWD/../coq/" make
  ) || exit 1
}

function setup-compcert {
    check-and-clone\
      "compcert" "git@github.com:AbsInt/CompCert.git"\
      "47f63df0a43209570de224f28cf53da6a758df16"
    {
        set -euv
        cd compcert
        ./configure x86_64-linux
        make -j `nproc`
    }
}

setup-coq
setup-coq-serapi
setup-compcert

#!/usr/bin/env bash
set -euv

function check-and-clone {
    if [ ! -d $1 ]; then
        git clone $2
    fi
    (cd $1 && git fetch && git checkout $3) || exit 1
}

check-and-clone\
    "coq" "https://github.com/coq/coq.git"\
    "9d423562a5f83563198f3141500af4c97103c2bf"

check-and-clone\
    "coq-serapi" "https://github.com/ejgallego/coq-serapi.git"\
    "352c255a6e7a66268197cd7c04b6dc47b3bb7536"

if [ -f /etc/NIXOS ]; then
  echo "Just run nix-shell"
else
  opam switch 4.04.0
  # For Coq:
  opam install menhir
  # For SerAPI:
  opam install ocamlfind ppx_import cmdliner core_kernel sexplib ppx_sexp_conv camlp5
  eval `opam config env`
fi

(cd coq; make -j `nproc`)

(cd coq-serapi; SERAPI_COQ_HOME="$PWD/../coq/" make)

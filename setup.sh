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

(cd coq; make -j `nproc`)

opam install ocamlfind ppx_import cmdliner core_kernel sexplib ppx_sexp_conv camlp5

SERAPI_COQ_HOME="$PWD/coq/" cd coq-serapi; make

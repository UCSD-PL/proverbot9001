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
    opam switch 4.03.0
    # For Coq:
    opam install menhir
    # For SerAPI:
    opam install ocamlfind ppx_deriving ppx_import cmdliner core_kernel sexplib ppx_sexp_conv camlp5
    eval `opam config env`
    pip3 install --user sexpdata
    pip3 install --user yattag
    pip3 install --user torch torchvision
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
        "coq-serapi" "https://github.com/Ptival/coq-serapi.git"\
        "601ad4f8baee98d025b8157c344d6b6155280930"
    (
        set -euv
        cd coq-serapi
        SERAPI_COQ_HOME="$PWD/../coq/" make
    ) || exit 1
}

function setup-coq-menhir {
    check-and-clone\
        "coq-menhirlib" "https://gitlab.inria.fr/fpottier/coq-menhirlib.git"\
        "b3a8229fa967a0185560f4741110f71f3b414de7"
    (
        set -euv
        cd coq-menhirlib
        export PATH="$PWD/../coq/bin:$PATH"
        make && make install
    )
}

# function setup-compcert {
#     check-and-clone\
#         "CompCert" "https://github.com/AbsInt/CompCert.git"\
#         "47f63df0a43209570de224f28cf53da6a758df16"
#     (
#         set -euv
#         cd CompCert
#         if [[ ! -f "Makefile.config" ]]; then
#             PATH="$PWD/../coq/bin:$PATH" ./configure x86_64-linux
#         fi
#         PATH="$PWD/../coq/bin:$PATH" make -j `nproc`
#     ) || exit 1
# }

function setup-software-foundation {
    check-and-clone\
        "software-foundations"\
        "https://github.com/Liby99/software-foundations.git"\
        "99a3f3d1e2526f2f89028605b1b441076d9c2838"
    (
        set -euv
        cd software-foundations
        PATH="$PWD/../coq/bin:$PATH" make
    ) || exit 1
}

setup-coq
# setup-coq-serapi
setup-coq-menhir
# setup-compcert
setup-software-foundation

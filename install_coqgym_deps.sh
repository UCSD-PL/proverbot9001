#!/usr/bin/env bash

if ! command -v ruby &> /dev/null
then
    # First, install Ruby, as that is for some reason required to build
    # the "system" project
    git clone https://github.com/rbenv/ruby-build.git ~/ruby-build
    mkdir -p ~/.local
    PREFIX=~/.local ~/ruby-build/install.sh
    ~/.local/ruby-build 3.1.2 ~/.local/
fi

git submodule init && git submodule update

# Sync opam state to local
rsync -av --delete $HOME/.opam.dir/ /tmp/${USER}_dot_opam | tqdm --desc="Reading shared opam state" > /dev/null

# Create the 8.10 switch
opam switch create coq-8.10 4.07.1
eval $(opam env --switch=coq-8.10 --set-switch)
opam pin add -y coq 8.10.2

# Install dependency packages for 8.10
opam repo add coq-extra-dev https://coq.inria.fr/opam/extra-dev
opam repo add coq-released https://coq.inria.fr/opam/released
opam repo add psl-opam-repository https://github.com/uds-psl/psl-opam-repository.git
opam install -y coq-serapi \
     coq-struct-tact \
     coq-inf-seq-ext \
     coq-cheerios \
     coq-verdi \
     coq-smpl \
     coq-int-map \
     coq-pocklington \
     coq-mathcomp-ssreflect coq-mathcomp-bigenough coq-mathcomp-algebra\
     coq-fcsl-pcm \
     coq-ext-lib \
     coq-simple-io \
     coq-list-string \
     coq-error-handlers \
     coq-function-ninjas \
     coq-algebra \
     coq-zorns-lemma

opam pin -y add menhir 20190626
# coq-equations seems to rely on ocamlfind for it's build, but doesn't
# list it as a dependency, so opam sometimes tries to install
# coq-equations before ocamlfind. Splitting this into a separate
# install call prevents that.
opam install -y coq-equations \
     coq-metacoq coq-metacoq-checker coq-metacoq-template

# Metalib doesn't install properly through opam unless we use a
# specific commit.
(cd coq-projects/metalib && opam install .)

(cd coq-projects/lin-alg && make "$@" && make install)

# Install the psl base-library from source
mkdir -p deps
git clone -b coq-8.10 git@github.com:uds-psl/base-library.git deps/base-library
(cd deps/base-library && make "$@" && make install)

git clone git@github.com:davidnowak/bellantonicook.git deps/bellantonicook
(cd deps/bellantonicook && make "$@" && make install)

# Create the coq 8.12 switch
opam switch create coq-8.12 4.07.1
eval $(opam env --switch=coq-8.12 --set-switch)
opam pin add -y coq 8.12.2

# Install the packages that can be installed directly through opam
opam repo add coq-released https://coq.inria.fr/opam/released
opam repo add coq-extra-dev https://coq.inria.fr/opam/extra-dev
opam install -y coq-serapi \
     coq-smpl=8.12 coq-metacoq-template coq-metacoq-checker \
     coq-equations \
     coq-mathcomp-ssreflect coq-mathcomp-algebra coq-mathcomp-field \
     menhir

# Install some coqgym deps that don't have the right versions in their
# official opam packages
git clone git@github.com:uwplse/StructTact.git deps/StructTact
(cd deps/StructTact && opam install -y . )
git clone git@github.com:DistributedComponents/InfSeqExt.git deps/InfSeqExt
(cd deps/InfSeqExt && opam install -y . )
# Cheerios has its own issues
git clone git@github.com:uwplse/cheerios.git deps/cheerios
(cd deps/cheerios && opam install -y --ignore-constraints-on=coq . )
(cd coq-projects/verdi && opam install -y --ignore-constraints-on=coq . )
(cd coq-projects/fcsl-pcm && make "$@" && make install)

# Finally, sync the opam state back to global
rsync -av --delete /tmp/${USER}_dot_opam/ $HOME/.opam.dir | tqdm --desc="Writing shared opam state" > /dev/null

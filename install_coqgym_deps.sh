#!/usr/bin/env bash

# First, install Ruby, as that is for some reason required to build
# the "system" project
git clone https://github.com/rbenv/ruby-build.git
mkdir -p ~/.local
PREFIX=~/.local ./ruby-build/install.sh
~/.local/ruby-build 3.1.2 ~/.local/

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
     coq-metacoq coq-metacoq-checker coq-metacoq-template \
     coq-smpl \
     coq-int-map \
     coq-pocklington \
     coq-mathcomp-ssreflect coq-mathcomp-bigenough coq-mathcomp-algebra\
     coq-fcsl-pcm \
     coq-ext-lib \
     coq-simple-io
opam pin -y add menhir 20190626
# coq-equations seems to rely on ocamlfind for it's build, but doesn't
# list it as a dependency, so opam sometimes tries to install
# coq-equations before ocamlfind. Splitting this into a separate
# install call prevents that.
opam install -y coq-equations

# Metalib doesn't install properly through opam unless we use a
# specific commit.
(cd coq-projects/metalib && opam install .)

# Install the psl base-library from source
mkdir deps
git clone -b coq-8.10 git@github.com:uds-psl/base-library.git deps/base-library
(cd deps/base-library && make "$@" && make install)

# Create the coq 8.12 switch
opam switch create coq-8.12 4.07.1
eval $(opam env --switch=coq-8.12 --set-switch)
opam pin add -y coq 8.12.2

# Install some coqgym deps that don't have the right versions in their
# official opam packages
git clone git@github.com:uwplse/StructTact.git deps/StructTact
(cd deps/StructTact && opam install .)
git clone git@github.com:DistributedComponents/InfSeqExt.git deps/InfSeqExt
(cd deps/InfSeqExt && opam install .)
git clone git@github.com:uwplse/cheerios.git deps/cheerios
(cd deps/cheerios && opam install .)

# Same with verdi, but modify it's existing opam file to allow 8.12 (I was told by one of the developers that this is okay https://github.com/uwplse/verdi-raft/issues/89)
(cd coq-projects/verdi &&
     sed -i 's/< "8.12~"/<= "8.12~"/' coq-projects/verdi/coq-verdi.opam &&
     opam install .)

# Install the packages that can be installed directly through opam
opam repo add coq-released https://coq.inria.fr/opam/released 
opam repo add coq-extra-dev https://coq.inria.fr/opam/extra-dev
opam install -y coq-serapi \
     coq-smpl coq-metacoq-template \
     coq-equations \
     coq-mathcomp-ssreflect coq-mathcomp-algebra coq-mathcomp-field \
     coq-fcsl-pcm \
     menhir


# Finally, sync the opam state back to global
rsync -av --delete /tmp/${USER}_dot_opam/ $HOME/.opam.dir | tqdm --desc="Writing shared opam state" > /dev/null

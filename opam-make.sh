# source $HOME/work/proverbot9001/prelude.sh
rsync -av --delete $HOME/.opam.dir/ /tmp/${USER}_dot_opam | tqdm --desc="Reading shared opam state" > /dev/null
make "$@"

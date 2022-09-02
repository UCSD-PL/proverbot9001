#!/usr/bin/env bash


INIT_CMD="~/opam-scripts/read-opam.sh"

NTHREADS=1
while getopts ":j:" opt; do
  case "$opt" in
    j)
      NTHREADS="${OPTARG}"
      ;;
  esac
done

# Make sure ruby is in the path
export PATH=$HOME/.local/bin:$PATH

git submodule init && git submodule update
for project in $(jq -r '.[].project_name' coqgym_projs_splits.json); do

    echo "#!/usr/bin/env bash" > coq-projects/$project/scrape.sh
    echo ${INIT_CMD} >> coq-projects/$project/scrape.sh

    SWITCH=$(jq -r ".[] | select(.project_name == \"$project\") | .switch" coqgym_projs_splits.json)
    echo "eval \"$(opam env --set-switch --switch=$SWITCH)\"" >> coq-projects/$project/scrape.sh
    FILES=$(echo $(jq -r ".[] | select(.project_name == \"$project\") | (.test_files[], .train_files[])" coqgym_projs_splits.json))

    echo "./src/scrape.py -c --prelude=./coq-projects/$project $@ ${FILES} > /dev/null" >> coq-projects/$project/scrape.sh
    chmod u+x coq-projects/$project/scrape.sh
    set -x
    ./src/sbatch-retry.sh --cpus-per-task=${NTHREADS} -o "coq-projects/$project/scrape-output.out" "coq-projects/$project/scrape.sh"
    set +x
done

#!/usr/bin/env bash


INIT_CMD="module load opam"

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
for project in $(jq -r '.[].project_name' coqgym_test_split.json); do
    SWITCH=$(jq -r ".[] | select(.project_name == \"$project\") | .switch" coqgym_test_split.json)
    FILES=$(echo $(jq -r ".[] | select(.project_name == \"$project\") | (.test_files[], .train_files[])" coqgym_test_split.json))
    if [ -z "$FILES" ]; then
        continue
    fi

    echo "#!/usr/bin/env bash" > coq-projects/$project/scrape.sh
    echo ${INIT_CMD} >> coq-projects/$project/scrape.sh

    echo "eval \"$(opam env --set-switch --switch=$SWITCH)\"" >> coq-projects/$project/scrape.sh

    echo "./src/scrape.py -c --prelude=./coq-projects/$project $@ ${FILES} > /dev/null" >> coq-projects/$project/scrape.sh
    chmod u+x coq-projects/$project/scrape.sh
    set -x
    ./src/sbatch-retry.sh --cpus-per-task=${NTHREADS} -o "coq-projects/$project/scrape-test-output.out" "coq-projects/$project/scrape.sh"
    set +x
done

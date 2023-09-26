#!/usr/bin/env bash


# INIT_CMD="~/opam-scripts/read-opam.sh"
INIT_CMD=""

NTHREADS=1
while getopts ":j:" opt; do
  case "$opt" in
    j)
      NTHREADS="${OPTARG}"
      shift
      ;;
  esac
done

# Make sure ruby is in the path
export PATH=$HOME/.local/bin:$PATH

TARGETS=${@:-$(jq -r '.[].project_name' coqgym_projs_splits.json)}

for project in $TARGETS; do
    SWITCH=$(jq -r ".[] | select(.project_name == \"$project\") | .switch" coqgym_projs_splits.json)
    if $(jq -e ".[] | select(.project_name == \"$project\") | has(\"prelude\")" \
         coqgym_projs_splits.json); then
        PRELUDE=$(jq -r ".[] | select(.project_name == \"$project\") | .prelude" coqgym_projs_splits.json)
        echo "Prelude is $PRELUDE"
    else
        PRELUDE=""
        echo "No prelude"
    fi
    FILES=$(echo $(jq -r ".[] | select(.project_name == \"$project\") | (.test_files[], .train_files[])" coqgym_projs_splits.json))
    if [ -z "$FILES" ]; then
        continue
    fi

    echo "#!/usr/bin/env bash" > coq-projects/$project/scrape.sh
    echo ${INIT_CMD} >> coq-projects/$project/scrape.sh

    echo "eval \"$(opam env --set-switch --switch=$SWITCH)\"" >> coq-projects/$project/scrape.sh

    echo "./src/scrape.py -c --prelude=./coq-projects/$project/$PRELUDE ${FILES} > /dev/null" >> coq-projects/$project/scrape.sh
    chmod u+x coq-projects/$project/scrape.sh
    set -x
    ./src/sbatch-retry.sh --cpus-per-task=${NTHREADS} -o "coq-projects/$project/scrape-output.out" "coq-projects/$project/scrape.sh"
    set +x
done

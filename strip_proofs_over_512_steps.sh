#!/usr/bin/env bash
[[ $# -ne 2 ]] && echo "Wrong number of arguments $# (2 expected)" && exit 1
[ -d $2 ] && echo "Output dir already exists" && exit 1
rsync -avzz --include="*/" --include="*-proofs.txt" --exclude="*" $1 $2
for proofsfile in $(find $2 -name "*-proofs.txt"); do
    cat $proofsfile | jq -c ".[1].status = if .[1].steps_taken > 512 and .[1].status == \"SUCCESS\" then \"INCOMPLETE\" else .[1].status end" > $proofsfile.fixed
    mv $proofsfile{.fixed,}
done

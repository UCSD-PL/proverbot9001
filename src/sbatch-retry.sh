#!/usr/bin/env bash

CUR_DIR=/work/pi_brun_umass_edu/asanchezster/proverbot9001/src/
BACKOFF_AMOUNT=0.001
SFLAGS="-u $USER -h"
while getopts ":J:" opt; do
    case "$opt" in
      J)
        SFLAGS+=" -n ${OPTARG}"
        ;;
      ?)
        continue
        ;;
    esac
done
JOBS_BEFORE=$($CUR_DIR/squeue-retry.sh $SFLAGS)
while
    sbatch "$@"
    /usr/bin/env sleep $BACKOFF_AMOUNT
    JOBS_AFTER=$($CUR_DIR/squeue-retry.sh $SFLAGS)
    diff <(echo "$JOBS_BEFORE" | awk '{print $1}' | sort) \
         <(echo "$JOBS_AFTER"  | awk '{print $1}' | sort) | grep -q "> "
    (( $? != 0 ))
do
    echo "Submission failed, retrying with delay ${BACKOFF_AMOUNT}s..." >&2
    BACKOFF_AMOUNT=$(awk "BEGIN {print ($BACKOFF_AMOUNT * 2)}")
    JOBS_BEFORE="$JOBS_AFTER"
done

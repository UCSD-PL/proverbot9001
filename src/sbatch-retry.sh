#!/usr/bin/env bash

CUR_DIR=$HOME/work/search-diversity-proverbot/proverbot9001/src/
BACKOFF_AMOUNT=0.001
MAX_BACKOFF=60
MAX_QUEUE=600
MAX_RETRIES=10
LOG_FILE="submitted_jobs.log"
SFLAGS="-u $USER -h"
while getopts ":J:" opt; do
    case "$opt" in
      J)
        SFLAGS+=" -n ${OPTARG}"
        ;;
    esac
done
RETRIES=0

while true; do
    # Throttle if too many jobs already in queue
    CURRENT_QUEUE=$($CUR_DIR/squeue-retry.sh -u $USER | wc -l)
    if [[ $CURRENT_QUEUE -ge $MAX_QUEUE ]]; then
        echo "Queue too deep ($CURRENT_QUEUE). Waiting..." >&2
        sleep 60
        continue
    fi

    # Try to submit the job
    OUTPUT=$(sbatch "$@" 2>&1)
    STATUS=$?
    echo "$OUTPUT"

    # Exit if success
    if [[ $STATUS -eq 0 && "$OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
        echo "$JOB_ID" >> "$LOG_FILE"
        break
    fi

    # Retry logic
    echo "Submission failed: $OUTPUT" >&2
    ((RETRIES++))
    if [[ $RETRIES -ge $MAX_RETRIES ]]; then
        echo "Too many retries. Giving up." >&2
        exit 1
    fi

    echo "Retrying after ${BACKOFF_AMOUNT}s..." >&2
    sleep $BACKOFF_AMOUNT
    BACKOFF_AMOUNT=$(awk -v b="$BACKOFF_AMOUNT" -v max="$MAX_BACKOFF" 'BEGIN {print (b*2 > max ? max : b*2)}')
done


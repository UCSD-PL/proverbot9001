#!/usr/bin/env bash

CACHE_FILE="squeue_cache.txt"
TTL=2  # seconds
MAX_RETRIES=10
BACKOFF_AMOUNT=0.5
RETRIES=0

# Check if cache exists and is fresh
if [ -f "$CACHE_FILE" ] && [ $(( $(date +%s) - $(stat -c %Y "$CACHE_FILE") )) -lt $TTL ]; then
    cat "$CACHE_FILE"
    exit 0
fi

# Try squeue with retries and backoff
while ! squeue "$@" > "$CACHE_FILE" 2>/dev/null; do
    ((RETRIES++))
    if [[ $RETRIES -ge $MAX_RETRIES ]]; then
        echo "squeue failed after $MAX_RETRIES retries" >&2
        exit 1
    fi
    sleep $BACKOFF_AMOUNT
    BACKOFF_AMOUNT=$(awk -v b="$BACKOFF_AMOUNT" 'BEGIN {print (b < 5.0 ? b*2 : 5.0)}')
done

cat "$CACHE_FILE"


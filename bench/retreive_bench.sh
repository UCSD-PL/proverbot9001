#!/usr/bin/env bash

set -e
[ "$#" -ge 2 ] || (echo "at least 2 argument required, $# provided" && exit 1)

PROJECT_DIR=$1
WEIGHTS_POSTFIX=$2
PROJECT_NAME=$(basename "$PROJECT_DIR")
shift 2

rsync -avzz $PROJECT_DIR/search-report-$WEIGHTS_POSTFIX .
mv search-report-$WEIGHTS_POSTFIX{,-$PROJECT_NAME}
firefox search-report-$WEIGHTS_POSTFIX-$PROJECT_NAME/report.html

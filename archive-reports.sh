#!/usr/bin/env bash

set -e

ARCHIVE_DIR="/project/pi_brun_umass_edu/$USER/proverbot9001-archives"
mkdir -p $ARCHIVE_DIR

# Try to get into the right directory for executing this
if [ ! -z "$SLURM_SUBMIT_DIR" ]; then
  # If we're a slurm job, this script was moved before executing, but we can
  # run from the directory where the user submitted the job, which is hopefully
  # right.
  cd $SLURM_SUBMIT_DIR
else
  # Otherwise, we can use the location of this script to run from
  src="${BASH_SOURCE[0]}"
  while [ -L "$src" ]; do
    dir="$(cd -P "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    [[ $src != /* ]] && src="$dir/$src"
  done
  MYDIR="$(cd -P "$(dirname "$src")" && pwd)"
  cd $MYDIR
fi

REPORT_TRIGGER_FILES=$(find . -name "workers_scheduled.txt" -or -name "learner_scheduled.txt")
if [ ! -z $REPORT_TRIGGER_FILES ]; then
  REPORT_TRIGGERED_DIRS=$(echo $REPORT_TRIGGER_FILES | dirname)
else
  REPORT_TRIGGERED_DIRS=""
fi

# Archive everything matching this find pattern.
REPORTS=$(echo $REPORT_TRIGGERED_DIRS $(find -type d -name "*.json.d" ) | sort | uniq)
for report in $REPORTS; do
  SIZE=$(du -sh $report)
  echo "Compressing "$report" ("$SIZE")..."
  tar czf $report{.tar.gz,}
  rm -r $report
  echo "Moving it to "$ARCHIVE_DIR"..."
  mv {,$ARCHIVE_DIR/}$report.tar.gz
done


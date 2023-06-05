#!/bin/bash

log_bash_history()
{
  local rc=$?
  local cwd=$(pwd)
  local SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  local PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd )"

  if [[ "$cwd" == "$PROJECT_ROOT"* ]]; then
    [[ "$(history 1)" =~ ^\ *[0-9]+\ +([^\ ]+\ [^\ ]+)\ +(.*)$ ]]
    if [[ ${BASH_REMATCH[1]} != "${BASH_LAST_EXECUTED_COMMAND}" ]]; then
      local LOG_FILE="$PROJECT_ROOT/.bash_history"
      echo "$(date "+%Y-%m-%d.%H:%M:%S") $(hostname -s) [$$]: ${BASH_COMMAND}" >> "$LOG_FILE"
      export BASH_LAST_EXECUTED_COMMAND="${BASH_REMATCH[1]}"
    fi
  fi
  return $rc
}

export PROMPT_COMMAND="log_bash_history"

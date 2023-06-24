#!/bin/bash

echo "tracking history"

# I'm nervous about how bash's dynamic scoping affects this
# it works if it's lexical scoping, which I'm more used to
SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd )"

_bash_history_append()
{
  local LOG_FILE="$PROJECT_ROOT/.bash_history"
  history -a $LOG_FILE
}

export PROMPT_COMMAND="_bash_history_append; $PROMPT_COMMAND"

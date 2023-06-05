#!/usr/local/bin/fish

echo "tracking history"

function log_fish_persistent_history --on-event fish_preexec
    set -l cwd (pwd)
    set -l script_directory (realpath (dirname (status --current-filename)))
    set -l project_root (realpath "$script_directory/..")
    if string match -q "$project_root*" "$cwd"
        set -l log_file "$project_root/.fish_history"
        echo (date "+%Y-%m-%d.%H:%M:%S") (hostname) [$fish_pid]: $argv >> $log_file
    end
end

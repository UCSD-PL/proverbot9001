#!/bin/bash

# Function to create and configure a tmux session
function create_tmux_session() {
    local session_name="$1"
    local command_to_run="$2"

    tmux new-session -d -s "$session_name"
    tmux send-keys "$command_to_run" C-m
}

# Specify the directory and command to run for each session
 #"wandb agent --count 10000 dylanszzhang/sweep-jul19/evquwck0"
#https://wandb.ai/dylanszzhang/sweep-jul19/sweeps/4wkfd7tp
# Start tmux sessions
for i in {1..6}; do
    session_name="davinci_worker-1-$i"
    create_tmux_session "$session_name" "bash run_two_step_labelings-step2-1-$i.sh"
done


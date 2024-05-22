#!/usr/bin/env python3

import argparse
import json
import sys
from coq_serapy import ProofContext
from search_results import SearchResult, TacticInteraction
from typing import List

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    with open(args.input_file, 'r') if args.input_file != "-" else sys.stdin as f:
        results = [(file_spec, SearchResult.from_dict(result_data))
                  for l in f for file_spec, result_data in (json.loads(l),)]

    new_results = [(file_spec, add_goal_control(result))
                   for file_spec, result in results]
    f = open(args.output_file, 'w') if args.output_file != "-" else sys.stdout
    for file_spec, result in new_results:
        print(json.dumps((file_spec, result.to_dict())), file=f)

def add_goal_control(result: SearchResult) -> SearchResult:
    new_commands: List[TacticInteraction] = []
    assert result.commands is not None
    num_goals_stack: List[int] = []
    prev_num_bg_goals = 0
    for command in result.commands[1:]:
        num_bg_goals = len(command.context_before.bg_goals)
        previous_context = command.context_before
        if num_bg_goals < prev_num_bg_goals:
            # Before we close a completed subgoal, all our goals are
            # in the background
            reconstructed_context = ProofContext(
                [], previous_context.fg_goals + previous_context.bg_goals,
                previous_context.shelved_goals, previous_context.given_up_goals)
            new_commands.append(TacticInteraction("}", reconstructed_context))
            ppInter(new_commands[-1])
            num_goals_stack[-1] -= 1
            while num_goals_stack[-1] == 0:
                num_goals_stack.pop()
                new_commands.append(TacticInteraction("}", reconstructed_context))
                ppInter(new_commands[-1])
                num_goals_stack[-1] -= 1
            if len(command.context_before.all_goals) != 0:
                # When we're opening a new bracket after closing a
                # bunch of brackets, the number of goals that are put
                # in the background is one minus the number left on
                # the subgoals stack from when we made this split in
                # the first place.
                reconstructed_context = ProofContext(
                    previous_context.fg_goals +
                    previous_context.bg_goals[:num_goals_stack[-1]-1],
                    previous_context.bg_goals[num_goals_stack[-1]-1:],
                    previous_context.shelved_goals,
                    previous_context.given_up_goals)
                new_commands.append(TacticInteraction("{", reconstructed_context))
                ppInter(new_commands[-1])
        if num_bg_goals > prev_num_bg_goals:
            num_goals_in_split = len(previous_context.bg_goals) - prev_num_bg_goals + 1
            num_goals_stack.append(num_goals_in_split)
            assert len(previous_context.fg_goals) == 1
            # Before we open a new bracket, the goals we just created
            # are in the foreground. But the first one will stay in
            # the foreground after, so move n-1 goals from the
            # background to the foreground.
            reconstructed_context = ProofContext(
                previous_context.fg_goals + previous_context.bg_goals[:num_goals_in_split-1],
                previous_context.bg_goals[num_goals_in_split-1:],
                previous_context.shelved_goals,
                previous_context.given_up_goals)
            new_commands.append(TacticInteraction("{", reconstructed_context))
            ppInter(new_commands[-1])
        if len(previous_context.fg_goals) == 0:
            while len(num_goals_stack) > 0:
                new_commands.append(TacticInteraction("}", previous_context))
                ppInter(new_commands[-1])
                num_goals_stack.pop()
        prev_num_bg_goals = num_bg_goals
        new_commands.append(command)
        ppInter(new_commands[-1])
    return SearchResult(result.status, result.context_lemmas,
                        new_commands, result.steps_taken)

def eprint(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stderr)

# Uncomment this function for debugging
def ppInter(t: TacticInteraction) -> None:
    # eprint(f"{len(t.context_before.fg_goals)} goals in foreground, "
    #        f"{len(t.context_before.bg_goals)} goals in background")
    # eprint(t.tactic)
    pass

if __name__ == "__main__":
    main()

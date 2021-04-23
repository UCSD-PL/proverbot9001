import argparse
import re
from typing import List, IO, Tuple

from pathlib_revised import Path2
from tqdm import tqdm

import coq_serapy
import linearize_semicolons


def generate_synthetic_lemmas(coq: coq_serapy.SerapiInstance,
                              lemma_idx: int,
                              lemma_stmt: str,
                              local_vars: List[str],
                              proof_commands: List[str], f: IO[str]):
    def write(s: str) -> None:
        print(s, file=f)
    break_after = False
    for cmd_idx in range(len(proof_commands)):
        if proof_commands[cmd_idx].startswith("intro"):
            continue
        cur_cmd = proof_commands[cmd_idx]
        assert coq.proof_context
        is_goal_open = re.match(r"\s*(?:\d+\s*:)?\s*[{]\s*", cur_cmd)
        is_goal_close = re.match(r"\s*[}]\s*", cur_cmd)
        if coq.count_fg_goals() > 1 and not is_goal_open:
            coq.run_stmt("{")
        before_state = coq.tactic_context([])
        if is_goal_open or is_goal_close:
            continue
        coq.run_stmt(cur_cmd)

        if not coq.proof_context or len(coq.proof_context.all_goals) == 0:
            after_goals = []
            break_after = True
        else:
            after_goals = coq.proof_context.fg_goals
        # NOTE: for now we're generating synth lemmas on anything that doesn't
        # manipulate the goal

        # for now only " h1 => g2 => g1 "

        before_hyps = before_state.hypotheses

        # This block ensures that if the induction tactic generalizes a
        # variable already in context, we'll put that variable generalized
        # in the subgoal hyp precedents.

        generalized_vars = []
        induction_match = re.match(r"\s*induction\s+(?P<var>\S)\.",
                                   cur_cmd)
        if induction_match:
            var = induction_match.group('var')
            for hyp in before_hyps:
                hyp_vars = [h.strip() for h in
                            coq_serapy.get_var_term_in_hyp(hyp).split(",")]
                if var in hyp_vars:
                    generalized_vars.append(
                        var + " : " +
                        coq_serapy.get_hyp_type(hyp))
                    break

        sec_name = "test_sec"
        write(f"Section {sec_name}.")
        for h in reversed(before_state.hypotheses):
            write(f"  Hypothesis {h}.")
        for gidx, goal in enumerate(after_goals):
            gname = f"test_goal{gidx}"

            new_hyps = generalized_vars + \
                list(reversed(
                    hyps_difference(goal.hypotheses, before_hyps)))
            gbody = ""

            for new_hyp in new_hyps:
                gbody += f"forall ({new_hyp}), "
            gbody += goal.goal

            write(f"  Hypothesis {gname}: {gbody}.")

        synth_lemma_name = f"synth_lemma_{cmd_idx}"
        write(f"  Lemma {synth_lemma_name}: {before_state.goal}.")
        write("Admitted.")
        write(f"End {sec_name}.")
        if break_after:
            break

    lemma_name = coq_serapy.lemma_name_from_statement(lemma_stmt)
    coq.run_stmt(f"Reset {lemma_name}.")
    coq.run_stmt(lemma_stmt)


def hyps_difference(hyps_base: List[str],
                    hyps_subtracted: List[str]) -> List[str]:
    result = []
    for hyp in hyps_base:
        vars_in_hyp = [name.strip() for name in
                       coq_serapy.get_var_term_in_hyp(hyp).split(",")]
        vars_left = [var for var in vars_in_hyp
                     if var not in
                     coq_serapy.get_vars_in_hyps(hyps_subtracted)]
        if len(vars_left) > 0:
            result.append(", ".join(vars_left) + " : " +
                          coq_serapy.get_hyp_type(hyp))
    return result


def generate_synthetic_file(args: argparse.Namespace,
                            filename: Path2,
                            proof_jobs: List[str]):
    local_vars = []

    def add_local_vars(cmds: List[str]) -> None:
        for cmd in cmds:
            variable_match = re.fullmatch(r"\s*Variable\s+(.*)\.\s*", cmd)
            if variable_match:
                local_vars.append(variable_match.group(1))
            let_match = re.match(r"\s*Let", cmd)
            if let_match:
                print(f"Adding {cmd} to local vars as "
                      f"{coq_serapy.let_to_hyp(cmd)}")
                local_vars.append(coq_serapy.let_to_hyp(cmd))

    synth_filename = args.prelude / Path2(str(filename.with_suffix(""))
                                          + '-synthetic.v')
    with synth_filename.open('w') as synth_f:
        pass

    coqargs = ["sertop", "--implicit"]
    proof_commands = linearize_semicolons.get_linearized(
        args, coqargs, 0, str(filename))
    with tqdm(desc='Processing proofs', total=len(proof_commands)) as bar:
        with coq_serapy.SerapiContext(coqargs,
                                      None,
                                      str(args.prelude)) as coq:
            coq.verbose = args.verbose
            rest_commands = proof_commands
            while True:
                rest_commands, run_commands = coq.run_into_next_proof(
                    rest_commands)
                add_local_vars(run_commands)
                bar.update(len(run_commands))
                with synth_filename.open('a') as synth_f:
                    for cmd in run_commands[:-1]:  # discard starting the proof
                        print(cmd, file=synth_f, end="")
                    if not coq.proof_context:
                        break
                    lemma_statement = run_commands[-1]
                    lemma_name = coq_serapy.lemma_name_from_statement(
                        lemma_statement)
                    if lemma_name in proof_jobs:
                        generate_synthetic_lemmas(coq,
                                                  proof_jobs.index(lemma_name),
                                                  lemma_statement,
                                                  local_vars,
                                                  rest_commands, synth_f)
                    rest_commands, run_commands = coq.finish_proof(
                        rest_commands)
                    bar.update(len(run_commands))
                    print(lemma_statement.strip(), file=synth_f)
                    for cmd in run_commands:
                        print(cmd, file=synth_f, end="")


def main():
    parser = argparse.ArgumentParser(
        description="Generate n-step synthetic proofs")

    parser.add_argument("filenames", nargs="+",
                        help="Proof file names to generate from",
                        type=Path2)

    parser.add_argument("--prelude", default=".", type=Path2)

    proofsGroup = parser.add_mutually_exclusive_group()
    proofsGroup.add_argument("--proof", default=None)
    proofsGroup.add_argument("--proofs-file", default=None)

    parser.add_argument(
        "--context-filter", dest="context_filter", type=str,
        default="(goal-args+hyp-args+rel-lemma-args)%maxargs:1%default")
    parser.add_argument("--linearizer-timeout",
                        type=int, default=(60 * 60 * 2))
    parser.add_argument("--progress", action='store_true')
    parser.add_argument("--verbose", "-v", action='count', default=0)

    args = parser.parse_args()

    if args.proof:
        proof_names = [args.proof]
    elif args.proofs_file:
        with open(args.proofs_file, 'r') as f:
            proof_names = [line.strip() for line in f]
    else:
        proof_names = None

    for idx, filename in enumerate(args.filenames):
        if proof_names:
            proof_jobs = proof_names
        else:
            proof_jobs = [coq_serapy.lemma_name_from_statement(stmt)
                          for filename, module, stmt in
                          get_proofs(args, (idx, filename))]
        generate_synthetic_file(args, filename, proof_jobs)


def get_proofs(args: argparse.Namespace,
               t: Tuple[int, str],
               include_proof_relevant: bool = False
               ) -> List[Tuple[str, str, str]]:
    idx, filename = t
    commands = coq_serapy.load_commands_preserve(
        args, idx, args.prelude / filename)
    return [(filename, module, cmd) for module, cmd in
            coq_serapy.lemmas_in_file(
                filename, commands, include_proof_relevant)]


if __name__ == "__main__":
    main()

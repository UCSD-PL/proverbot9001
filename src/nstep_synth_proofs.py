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

    def termify_hyp(hyp: str) -> str:
        return coq_serapy.get_var_term_in_hyp(
            hyp).replace(
                ",", " ") + " : " + \
                coq_serapy.get_hyp_type(hyp)
    break_after = False
    for cmd_idx in range(len(proof_commands)):
        cur_cmd = proof_commands[cmd_idx]
        assert coq.proof_context
        is_goal_open = re.match(r"\s*(?:\d+\s*:)?\s*[{]\s*", cur_cmd)
        is_goal_close = re.match(r"\s*[}]\s*", cur_cmd)
        is_proof_keyword = re.match(r"\s*Proof.*", cur_cmd)
        if coq.count_fg_goals() > 1 and not is_goal_open:
            coq.run_stmt("{")
        before_state = coq.tactic_context([])
        coq.run_stmt(cur_cmd)

        if not coq.proof_context or len(coq.proof_context.all_goals) == 0:
            # Get ready to terminate when we're at the last state
            after_goals = []
            break_after = True
        else:
            after_goals = coq.proof_context.fg_goals

        # Skip some proof handling commands that don't change the goal
        # Unfortunately there's no good way to handle evars
        if re.match(r".*\s+\?\w", before_state.goal, re.DOTALL) \
                or is_goal_open or is_goal_close or is_proof_keyword:
            if break_after:
                break
            else:
                continue
        if any([re.match(r".*\s+\?\w", goal.goal, re.DOTALL)
                for goal in after_goals]):
            if break_after:
                break
            else:
                continue
        # NOTE: for now we're generating synth lemmas on anything that doesn't
        # manipulate the goal

        # for now only " h1 => g2 => g1 "

        before_hyps = before_state.hypotheses

        # This block ensures that if the induction tactic generalizes a
        # variable already in context, we'll put that variable generalized
        # in the subgoal hyp precedents.

        generalized_vars = []
        generalize_match = re.match(r"\s*(induction|destruct)\s+(?P<var>\S)\.",
                                    cur_cmd)
        if generalize_match:
            var = generalize_match.group('var')
            for hyp in before_hyps:
                hyp_vars = [h.strip() for h in
                            coq_serapy.get_var_term_in_hyp(hyp).split(",")]
                if var in hyp_vars:
                    generalized_vars.append(
                        var + " : " +
                        coq_serapy.get_hyp_type(hyp))
                    break

        synth_lemma_name = f"synth_lemma_{lemma_idx}_{cmd_idx}"
        synth_lemma_stmt = f"Lemma {synth_lemma_name} "

        write(synth_lemma_stmt)

        for h in reversed(hyps_difference(before_state.hypotheses,
                                          local_vars)):
            write(f"  ({termify_hyp(h)})")
        num_valid_goals = 0
        for gidx, goal in enumerate(after_goals):
            num_valid_goals += 1
            gname = f"subgoal{gidx}"

            new_hyps = generalized_vars + \
                list(reversed(
                    hyps_difference(goal.hypotheses, before_hyps)))
            gbody = ""

            binders: List[str] = []
            for new_hyp in reversed(new_hyps):
                hyp_vars = coq_serapy.get_vars_in_hyps([new_hyp])
                hyp_vars_present = []
                for var in hyp_vars:
                    if re.search(rf'(^|\W){var}(\W|$)', goal.goal) or \
                            any([re.search(rf'(^|\W){var}(\W|$)', binder)
                                for binder in binders]):
                        hyp_vars_present.append(var)
                if len(hyp_vars_present) > 0:
                    binders.append(
                        f"forall ({' '.join(hyp_vars_present)} : "
                        f"{coq_serapy.get_hyp_type(new_hyp)}), ")
            for binder in reversed(binders):
                gbody += binder

            gbody += goal.goal

            write(f"  ({gname}: {gbody})")

        write(f"    : {before_state.goal}.")
        write("Proof.")
        cmd_base = cur_cmd.strip()[:-1]
        if num_valid_goals > 0:
            finisher = ";".join([f"try eapply subgoal{idx}" for idx in
                                 range(num_valid_goals)]) \
                                 + "; eauto."
        else:
            finisher = "eauto."
        proof = f"  {cmd_base}; {finisher}"
        write(proof)
        write("Qed.")
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
        vars_left = []
        for var in vars_in_hyp:
            already_exists = False
            for other_hyp in hyps_subtracted:
                if var in coq_serapy.get_vars_in_hyps([other_hyp]) \
                        and coq_serapy.get_hyp_type(hyp) == \
                        coq_serapy.get_hyp_type(other_hyp):
                    already_exists = True
                    break
            if not already_exists:
                vars_left.append(var)
        if len(vars_left) > 0:
            result.append(", ".join(vars_left) + " : " +
                          coq_serapy.get_hyp_type(hyp))
    return result


def remove_broken_proofs(args: argparse.Namespace,
                         filein: Path2, fileout: Path2) -> None:
    in_cmds = coq_serapy.load_commands(str(filein),
                                       progress_bar=args.progress)
    section_buffer: List[str] = []
    in_testsec = False
    is_broken = False
    with fileout.open('w') as outf:
        print("Opening serapi instance")
        with coq_serapy.SerapiContext(["sertop", "--implicit"], None,
                                      str(args.prelude)) as coq:
            coq.verbose = args.verbose
            for cmd in tqdm(in_cmds, total=len(in_cmds)):
                if cmd.strip() == "Section test_sec.":
                    assert not in_testsec
                    in_testsec = True
                    is_broken = False
                elif cmd.strip() == "End test_sec.":
                    assert in_testsec
                    in_testsec = False
                    if not is_broken:
                        for buf_cmd in section_buffer:
                            print(buf_cmd, file=outf, end="")
                        print(cmd, file=outf, end="")
                        coq.run_stmt(cmd)
                    continue

                if in_testsec:
                    if is_broken:
                        continue
                    section_buffer.append(cmd)
                    try:
                        coq.run_stmt(cmd)
                    except coq_serapy.SerapiException:
                        if coq.proof_context:
                            coq.run_stmt("Admitted.")
                        coq.run_stmt("End test_sec.")
                        is_broken = True
                else:
                    coq.run_stmt(cmd)
                    print(cmd, file=outf, end="")


def generate_synthetic_file(args: argparse.Namespace,
                            in_filename: Path2,
                            out_filename: Path2,
                            proof_jobs: List[str]):
    local_vars: List[List[str]] = [[]]

    def add_local_vars(cmds: List[str], coq: coq_serapy.SerapiInstance
                       ) -> None:
        for cmd in cmds:
            cmd = coq_serapy.kill_comments(cmd).strip()
            variable_match = re.fullmatch(
                r"\s*Variables?\s+(.*)\.\s*", cmd)
            if variable_match:
                var_part, type_part = variable_match.group(1).split(":", 1)
                first_var = var_part.split()[0].strip()
                desugared_type = coq.check_term(first_var).split(
                    ":", 1)[1].strip()
                var_hyp = var_part.replace(" ", ", ") + " :" + desugared_type
                local_vars[-1].append(var_hyp)
            let_match = re.match(r"\s*Let\s+([\w']*)\s+", cmd)
            if let_match:
                ident = let_match.group(1)
                desugared_type = coq.check_term(
                    ident).split(":", 1)[1].strip()
                local_vars[-1].append(ident + " : " + desugared_type)
            section_match = re.match(r"\s*Section", cmd)
            if section_match:
                local_vars.append([])
            end_match = re.match(r"\s*End", cmd)
            if end_match:
                local_vars.pop()

    with out_filename.open('w') as synth_f:
        pass

    coqargs = ["sertop", "--implicit"]
    proof_commands = linearize_semicolons.get_linearized(
        args, coqargs, 0, str(in_filename))
    with tqdm(desc='Processing proofs', total=len(proof_commands)) as bar:
        with coq_serapy.SerapiContext(coqargs,
                                      None,
                                      str(args.prelude)) as coq:
            coq.verbose = args.verbose
            rest_commands = proof_commands
            coq.run_stmt("Set Printing Implicit.")
            while True:
                rest_commands, run_commands = coq.run_into_next_proof(
                    rest_commands)
                add_local_vars(run_commands, coq)
                bar.update(len(run_commands))
                with out_filename.open('a') as synth_f:
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
                                                  [var for var_list
                                                   in local_vars
                                                   for var in var_list],
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

        synth_filename = args.prelude / Path2(str(filename.with_suffix(""))
                                              + '-synthetic.v')
        generate_synthetic_file(args, filename,
                                synth_filename,
                                proof_jobs)


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

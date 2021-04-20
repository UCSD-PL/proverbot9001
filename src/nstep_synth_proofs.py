import argparse
from typing import List, TextIO, Tuple

from pathlib_revised import Path2

import coq_serapy
import util


def generate_synthetic_lemmas(coq: coq_serapy.SerapiInstance, 
                            proof_commands: List[str], f: TextIO):
    for cmd_idx in range(len(proof_commands)):
        before_state, after_state = coq.tactic_context([]), coq.tactic_context([])
        # NOTE: for now we're generating synth lemmas on anything that doesn't manipulate the goal
        # for now only " h1 => g1 => g2 "

        pass

def generate_synthetic_file(args: argparse.Namespace,
                            filename: Path2,
                            proof_jobs: List[str]):
    synth_filename = filename.get_stem().with_suffix('-synthetic.v')
    with synth_filename.open('a') as synth_f: 
        pass

    proof_commands = coq_serapy.load_commands(filename)
    with coq_serapy.SerapiContext(["sertop", "--implicit",
                                   f"--topfile={filename}"],
                                  None,
                                  str(args.prelude)) as coq:
        rest_commands = proof_commands
        while True:
            rest_commands, run_commands = coq.run_into_next_proof(rest_commands)
            with synth_filename.open('a') as synth_f:
                for cmd in run_commands[:-1]:  # discard starting the proof
                    print(cmd, file=synth_f)
                if coq_serapy.lemma_name_from_statement(run_commands[-1]) in proof_jobs:
                    generate_synthetic_lemmas(coq, rest_commands, synth_f)
                rest_commands, run_commands = coq.finish_proof(rest_commands)
                for cmd in run_commands:
                    print(cmd, file=synth_f)


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

    args = parser.parse_args()

    if args.proof:
        proof_names = [args.proof]
    elif args.proofs_file:
        with open(args.proofs_file, 'r') as f:
            proof_names = [line.strip() for line in f]
    else:
        proof_names = None

    for filename in args.filenames:
        if proof_names:
            proof_jobs = proof_names
        else:
            proof_jobs = [coq_serapy.lemma_name_from_statement(stmt)
                          for filename, module, stmt in
                          get_proofs(args, filename)]
        generate_synthetic_file(args, filename, proof_jobs)


def get_proofs(args: argparse.Namespace,
               t: Tuple[int, str]) -> List[Tuple[str, str, str]]:
    idx, filename = t
    with util.silent():
        commands = coq_serapy.load_commands_preserve(
            args, idx, args.prelude / filename)
    return [(filename, module, cmd) for module, cmd in
            coq_serapy.lemmas_in_file(
                filename, commands, args.include_proof_relevant)]


if __name__ == "__main__":
    main()

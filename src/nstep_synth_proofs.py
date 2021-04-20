import argparse
from typing import List, Tuple

from pathlib_revised import Path2

import coq_serapy
import util


def generate_synthetic_file(args: argparse.Namespace,
                            filename: Path2,
                            proof_jobs: List[str]):
    with coq_serapy.SerapiContext(["sertop", "--implicit",
                                   f"--topfile={filename}"],
                                  None,
                                  str(args.prelude)) as coq:
        pass

    pass


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
        cmds = coq_serapy.load_commands_preserve(
            args, idx, args.prelude / filename)
    return [(filename, module, cmd) for module, cmd in
            coq_serapy.lemmas_in_file(
                filename, cmds, args.include_proof_relevant)]


if __name__ == "__main__":
    main()

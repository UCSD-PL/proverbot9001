#!/usr/bin/env python3
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import argparse
import util
import coq_serapy
import tactic_classes
import linearize_semicolons
from pathlib_revised import Path2
from tqdm import tqdm
from models.class_polyarg_predictor import ClassPolyargPredictor


def score_classes(args: argparse.Namespace, filename: Path2):
    predictor = ClassPolyargPredictor()
    tactics_seen = 0
    tactics_class_correct = 0

    coqargs = ["sertop", "--implicit"]
    proof_commands = linearize_semicolons.get_linearized(
        args, coqargs, 0, str(filename))
    with tqdm(desc='Processing proofs', total=len(proof_commands)) as bar:
        with coq_serapy.SerapiContext(coqargs,
                                      coq_serapy.
                                        get_module_from_filename(filename),
                                      str(args.prelude)) as coq:
            coq.verbose = args.verbose
            rest_commands = proof_commands
            while rest_commands:
                rest_commands, run_commands = coq.run_into_next_proof(rest_commands)
                while coq.proof_context:
                    tactic = rest_commands[0]
                    tactic_class = tactic_classes.getTacticClass(tactic)
                    predicted_classes = predictor.predictClasses(coq.tactic_context)
                    if predicted_classes[0] == tactic_class:
                        tactics_class_correct += 1
                    tactics_seen += 1
                    coq.run_stmt(tactic)
                    rest_commands = rest_commands[1:]
                    run_commands.append(tactic)
        
    print(f'Class Prediction Accuracy: {tactics_class_correct}/{tactics_seen}')

def main():
    parser = argparse.ArgumentParser(
        description="Produce an html report from attempting "
        "to complete proofs using Proverbot9001.")
    parser.add_argument("--prelude", default=".")
    parser.add_argument("--verbose", "-v", help="verbose output",
                        action="count", default=0)
    parser.add_argument("--progress", "-P", help="show progress of files",
                        action='store_true')
    parser.add_argument('filenames', help="proof file name (*.v)",
                        nargs='+', type=Path2)

    args = parser.parse_args()
    for filename in args.filenames:
        score_classes(args, filename)

if __name__=="__main__":
    main()
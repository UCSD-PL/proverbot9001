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
import sys
from pathlib import Path
from typing import List

from predict_tactic import loadPredictorByFile
from models.tactic_predictor import TacticPredictor
from coq_serapy.contexts import TacticContext


def get_predictor(parser: argparse.ArgumentParser,
                  args: argparse.Namespace) -> TacticPredictor:
    predictor: TacticPredictor
    predictor = loadPredictorByFile(args.weightsfile)
    return predictor


def input_list() -> List[str]:
    result = []
    line = input()
    while line.strip() != "":
        result.append(line.strip())
        line = input()

    return result


def predict(args: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Proverbot9001 interactive prediction model")
    parser.add_argument("weightsfile", default=None, type=Path)
    parser.add_argument("-k", "--num-predictions", default=5)
    parser.add_argument("--print-certainties", action='store_true')
    arg_values = parser.parse_args(args[:1])

    predictor = get_predictor(parser, arg_values)

    while True:
        print("Enter relevant lemmas, one per line, "
              "with type annotations (blank line to finish list):")
        rel_lemmas = input_list()

        print("Enter previous tactics in proof, one per line, "
              "(blank line to finish list):")
        prev_tactics = input_list()

        print("Enter hypotheses, one per line, "
              "(blank line to finish list):")
        hypotheses = input_list()

        print("Enter goal (blank to exit):")
        goal = input()

        if goal.strip() == "":
            print("Exiting...")
            break

        tac_context = TacticContext(rel_lemmas,
                                    prev_tactics,
                                    hypotheses,
                                    goal)

        predictions = predictor.predictKTactics(
            tac_context, arg_values.num_predictions)

        for prediction in predictions:
            if arg_values.print_certainties:
                print(f"Prediction: \"{prediction.prediction}\"; "
                      f"certainty: {prediction.certainty}")
            else:
                print(prediction.prediction)

    pass

if __name__ == "__main__":
    predict(sys.argv[1:])

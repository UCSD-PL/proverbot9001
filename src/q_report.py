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
import torch
from pathlib_revised import Path2

import dataloader
import predict_tactic
from models.features_q_estimator import FeaturesQEstimator
from format import TacticContext


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A model for testing the effectiveness of the q estimator")

    parser.add_argument("predictor_weights", type=Path2)
    parser.add_argument("estimator_weights", type=Path2)
    parser.add_argument("test_files", nargs="+", type=Path2)
    parser.add_argument("--num_predictions", default=16, type=int)
    args = parser.parse_args()
    q_report(args)


def q_report(args: argparse.Namespace) -> None:
    num_originally_correct = 0
    num_correct = 0
    num_top3 = 0
    num_total = 0
    num_possible = 0

    predictor = predict_tactic.loadPredictorByFile(args.predictor_weights)
    q_estimator_name, *saved  = \
        torch.load(args.estimator_weights)
    q_estimator = FeaturesQEstimator(0, 0, 0)
    q_estimator.load_saved_state(*saved)

    for filename in args.test_files:
        points = dataloader.scraped_tactics_from_file(str(filename) +
                                                      ".scrape", None)
        for point in points:
            context = TacticContext(point.relevant_lemmas,
                                    point.prev_tactics,
                                    point.prev_hyps,
                                    point.prev_goal)
            predictions = [p.prediction for p in
                           predictor.predictKTactics(context,
                                                     args.num_predictions)]
            q_choices = zip(q_estimator([(context, prediction)
                                         for prediction in predictions]),
                            predictions)
            ordered_actions = [p[1] for p
                               in sorted(q_choices,
                                         key=lambda q: q[0],
                                         reverse=True)]

            num_total += 1
            if point.tactic.strip() in predictions:
                num_possible += 1

            if ordered_actions[0] == point.tactic.strip():
                num_correct += 1

            if point.tactic.strip() in ordered_actions[:3]:
                num_top3 += 1

            if predictions[0] == point.tactic.strip():
                num_originally_correct += 1
            pass

    print(f"num_correct: {num_correct}")
    print(f"num_originally_correct: {num_originally_correct}")
    print(f"num_top3: {num_top3}")
    print(f"num_total: {num_total}")
    print(f"num_possible: {num_possible}")


if __name__ == "__main__":
    main()

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
import re
import serapi_instance
import dataloader
import random

def main(arg_list : List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="A module for exploring deep Q learning with proverbot9001")

    parser.add_argument("scrape_file")

    parser.add_argument("environment_file")
    parser.add_argument("environment_proof", default=None)

    parser.add_argument("--prelude", default=".")

    parser.add_argument("--predictor-weights", default=Path2("data/polyarg_weights.dat"),
                        type=Path2)
    parser.add_argument("--num-predictions", default=16, type=int)

    parser.add_argument("--buffer-size", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)

    parser.add_argument("--num-episodes", default=256)
    parser.add_argument("--episode-length", default=16)

    parser.add_argument("--learning-rate", default=0.5)

    args = parser.parse_args()

    reinforce(args)

def reinforce(args : argparse.Namespace) -> None:

    # Load the scraped (demonstrated) samples, the proof environment
    # commands, and the predictor
    replay_memory = assign_rewards(
        dataloader.load_reinforce_samples(args.scrape_file,
                                          args.buffer_size))
    env_commands = serapi_instance.load_commands_preserve(args, 0, args.environment_file)
    predictor = loadPredictorByFile(args.predictor_weights)

    q_estimator = FeaturesQEstimator()
    epsilon = 0.3

    with serapi_instance.SerapiContext(
            ["sertop", "--implicit"],
            serapi_instance.get_module_from_filename(args.environment_file),
            args.prelude) as coq:
        ## Get us to the correct proof context
        rest_commands, _ = coq.run_into_next_proof(env_commands)
        if args.environment_proof != None:
            while coq.cur_lemma_name != args.environment_proof:
                if not rest_commands:
                    eprint("Couldn't find lemma {args.environment_proof}! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, _ = coq.run_into_next_proof(env_commands)
        else:
            # Don't use lemmas without names (e.g. "Obligation")
            while coq.cur_lemma_name == "":
                if not rest_commands:
                    eprint("Couldn't find usable lemma! Exiting...")
                    return
                rest_commands, _ = coq.finish_proof(rest_commands)
                rest_commands, _ = coq.run_into_next_proof(env_commands)

        lemma_name = coq.cur_lemma_name

        for episode in range(args.num_episodes):
            for t in range(args.episode_length):
                predictions = predictor.predictKTactics(
                    coq.tactic_context(coq.local_lemmas[:-1]),
                    args.num_predictions)
                if random.random() < epsilon:
                    action = random.choice(predictions).prediction
                else:
                    q_choices = [(q_estimator(coq.tactic_context, prediction.prediction),
                                  prediction.prediction)
                                 for prediction in predictions]
                    action = max(q_choices, key=lambda q: q[0])[1]

                coq.run_stmt(action)

                transition_samples = sample_batch(replay_memory, args.batch_size)
                training_samples = [assign_scores(transition_sample)
                                    for transition_sample in transition_samples]

                q_estimator.train(args.learning_rate, training_samples)
                pass

            # Clean up episode
            coq.run_stmt("Admitted.")
            coq.run_stmt("Reset {lemma_name}.")

if __name__ == "__main__":
    main(sys.argv[1:])

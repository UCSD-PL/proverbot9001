import argparse
import json
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
import torch.optim.lr_scheduler as scheduler
from util import maybe_cuda, eprint

import predict_tactic
from models import features_polyarg_predictor 
from pathlib_revised import Path2
from typing import (cast, Sequence)
from rgraph import LabeledTransition
from models.polyarg_q_estimator import PolyargQEstimator
from models.features_q_estimator import FeaturesQEstimator
from models.q_estimator import QEstimator
from reinforce import assign_scores

def supervised_q(args: argparse.Namespace) -> None:
    replay_memory = []
    with open(args.tmp_file, 'r') as f:
        for line in f:
            replay_memory.append(LabeledTransition.from_dict(
                json.loads(line)))

    # Load the predictor
    predictor = cast(features_polyarg_predictor.FeaturesPolyargPredictor,
                     predict_tactic.loadPredictorByFile(
                         args.predictor_weights))

    q_estimator: QEstimator
    # Create an initial Q Estimator
    if args.estimator == "polyarg":
        q_estimator = PolyargQEstimator(args.learning_rate,
                                        args.batch_step,
                                        args.gamma,
                                        predictor)
    else:
        q_estimator = FeaturesQEstimator(args.learning_rate,
                                         args.batch_step,
                                         args.gamma)
    input_tensors = q_estimator.get_input_tensors()

    for epoch in range(args.num_epochs):
        training_samples = assign_scores(args,
                                    replay_memory,
                                    q_estimator,
                                    predictor)
        scores = [score for _,_,_, score in training_samples]
        batches: Sequence[Sequence[torch.Tensor]] = data.DataLoader(
            data.TensorDataset(*(input_tensors + [scores])),
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True, pin_memory=True,
            drop_last=True)


        epoch_loss = 0.
        for idx, batch in enumerate(batches):
            q_estimator.optimizer.zero_grad()
            word_features_batch, vec_features_batch, \
                expected_outputs_batch = batch
            outputs = q_estimator.model(word_features_batch,
                                    vec_features_batch)
            loss = q_estimator.criterion(
                outputs, maybe_cuda(expected_outputs_batch))
            loss.backward()
            q_estimator.optimizer.step()
            q_estimator.adjuster.step()
            q_estimator.total_batches += 1
            epoch_loss += loss.item()
            eprint(epoch_loss / len(batches),
                    guard=args.show_loss and epoch % 10 == 0
                    and idx == len(batches) - 1)
            eprint("Batch {}: Learning rate {:.12f}".format(
                q_estimator.total_batches,
                q_estimator.optimizer.param_groups[0]['lr']),
                    guard=args.show_loss and epoch % 10 == 0
                    and idx == len(batches) - 1)


        pass

    pass

def main():
    parser = \
        argparse.ArgumentParser()

    parser.add_argument("tmp_file")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-epochs", default=256, type=int)
    parser.add_argument("--predictor-weights",
                    default=Path2("data/polyarg-weights.dat"),
                    type=Path2)
    parser.add_argument("--estimator",
                    choices=["polyarg", "features"],
                    default="polyarg")
    parser.add_argument("--show-loss", action='store_true')


    args = parser.parse_args()

if __name__ == "__main__":
    main()

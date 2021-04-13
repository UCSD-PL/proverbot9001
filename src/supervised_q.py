import argparse
import json
import time
import torch
import torch.utils.data as data
from tqdm import tqdm
from util import maybe_cuda, eprint, print_time, timeSince

import predict_tactic
from models import features_polyarg_predictor
from pathlib_revised import Path2
from typing import (cast, Sequence)
from rgraph import LabeledTransition
# from models.polyarg_q_estimator import PolyargQEstimator
from models.features_q_estimator import FeaturesQEstimator
# from models.q_estimator import QEstimator
from reinforce import assign_scores


def supervised_q(args: argparse.Namespace) -> None:
    replay_memory = []
    with open(args.tmp_file, 'r') as f:
        for idx, line in enumerate(tqdm(f, desc="Loading data")):
            if args.max_tuples is not None and idx >= args.max_tuples:
                break
            replay_memory.append(LabeledTransition.from_dict(
                json.loads(line)))

    # Load the predictor
    predictor = cast(features_polyarg_predictor.FeaturesPolyargPredictor,
                     predict_tactic.loadPredictorByFile(
                         args.predictor_weights))

    # q_estimator: QEstimator
    # Create an initial Q Estimator
    # if args.estimator == "polyarg":
    #     q_estimator = PolyargQEstimator(args.learning_rate,
    #                                     args.batch_step,
    #                                     args.gamma,
    #                                     predictor)
    # else:
    q_estimator = FeaturesQEstimator(args.learning_rate,
                                     args.batch_step,
                                     args.gamma)
    if args.start_from:
        q_estimator_name, *saved = \
          torch.load(args.start_from)
        assert q_estimator_name == "features evaluator", \
            q_estimator_name
        q_estimator.load_saved_state(*saved)

    with print_time("Assigning scores"):
        training_samples = assign_scores(args,
                                         replay_memory,
                                         q_estimator,
                                         predictor)
    input_tensors = q_estimator.get_input_tensors(training_samples)

    training_start = time.time()
    for epoch in range(args.num_epochs):
        scores = torch.FloatTensor([score for _, _, _, score
                                    in training_samples])
        batches: Sequence[Sequence[torch.Tensor]] = data.DataLoader(
            data.TensorDataset(*(input_tensors + [scores])),
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True, pin_memory=True,
            drop_last=True)

        epoch_loss = 0.
        for idx, batch in enumerate(batches, start=1):
            q_estimator.optimizer.zero_grad()
            word_features_batch, vec_features_batch, \
                expected_outputs_batch = batch
            outputs = q_estimator.model(word_features_batch,
                                        vec_features_batch)
            loss = q_estimator.criterion(
                outputs, maybe_cuda(expected_outputs_batch))
            loss.backward()
            q_estimator.optimizer.step()
            q_estimator.total_batches += 1
            epoch_loss += loss.item()
            if idx % args.print_every == 0:
                items_processed = idx * args.batch_size + \
                    (epoch - 1) * len(replay_memory)
                progress = items_processed / (len(replay_memory) *
                                              args.num_epochs)
                eprint("{} ({:7} {:5.2f}%) {:.4f}"
                       .format(timeSince(training_start, progress),
                               items_processed, progress * 100,
                               epoch_loss * (len(batches) / idx)),
                       guard=args.show_loss)

        q_estimator.adjuster.step()
        eprint("Epoch {}: Learning rate {:.12f}".format(
            epoch,
            q_estimator.optimizer.param_groups[0]['lr']),
                guard=args.show_loss)

        training_samples = assign_scores(args,
                                         replay_memory,
                                         q_estimator,
                                         predictor)

        pass

    pass


def main():
    parser = \
        argparse.ArgumentParser()

    parser.add_argument("tmp_file")
    parser.add_argument("--predictor-weights",
                        default=Path2("data/polyarg-weights.dat"),
                        type=Path2)
    parser.add_argument("--estimator",
                        choices=["polyarg", "features"],
                        default="polyarg")
    parser.add_argument("--start-from", default=None, type=Path2)

    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-epochs", default=256, type=int)
    parser.add_argument("--learning-rate", default=0.02, type=float)
    parser.add_argument("--batch-step", default=50, type=int)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--show-loss", action='store_true')
    parser.add_argument("--print-every", dest="print_every",
                        type=int, default=5)

    parser.add_argument("--max-term-length", default=512, type=int)
    parser.add_argument("--num-predictions", default=16, type=int)
    parser.add_argument("--max-tuples", default=None, type=int)

    parser.add_argument("--time-discount", default=0.9, type=float)

    args = parser.parse_args()
    supervised_q(args)


if __name__ == "__main__":
    main()

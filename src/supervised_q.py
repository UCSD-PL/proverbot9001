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
from models.polyarg_q_estimator import PolyargQEstimator
from models.features_q_estimator import FeaturesQEstimator
from models.q_estimator import QEstimator
from reinforce import assign_scores


def supervised_q(args: argparse.Namespace) -> None:
    replay_memory = []
    with open(args.tmp_file, 'r') as f:
        for idx, line in enumerate(tqdm(f, desc="Loading data")):
            replay_memory.append(LabeledTransition.from_dict(
                json.loads(line)))
    if args.max_tuples is not None:
        replay_memory = replay_memory[-args.max_tuples:]

    # Load the predictor
    predictor = cast(features_polyarg_predictor.FeaturesPolyargPredictor,
                     predict_tactic.loadPredictorByFile(
                         args.predictor_weights))

    q_estimator: QEstimator
    # Create an initial Q Estimator
    if args.estimator == "polyarg":
        q_estimator = PolyargQEstimator(args.learning_rate,
                                        args.epoch_step,
                                        args.gamma,
                                        predictor)
    else:
        q_estimator = FeaturesQEstimator(args.learning_rate,
                                         args.epoch_step,
                                         args.gamma)
    if args.start_from:
        q_estimator_name, *saved = \
          torch.load(args.start_from)
        if args.estimator == "polyarg":
            assert q_estimator_name == "polyarg evaluator", \
                q_estimator_name
        else:
            assert q_estimator_name == "features evaluator", \
                q_estimator_name
        q_estimator.load_saved_state(*saved)

    training_start = time.time()
    training_samples = assign_scores(args,
                                     q_estimator,
                                     predictor,
                                     replay_memory,
                                     progress=True)
    input_tensors = q_estimator.get_input_tensors(training_samples)
    rescore_lr = args.learning_rate

    for epoch in range(1, args.num_epochs+1):
        scores = torch.FloatTensor([score for _, _, _, score
                                    in training_samples])
        batches: Sequence[Sequence[torch.Tensor]] = data.DataLoader(
            data.TensorDataset(*(input_tensors + [scores])),
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True, pin_memory=True,
            drop_last=True)

        epoch_loss = 0.
        eprint("Epoch {}: Learning rate {:.12f}".format(
            epoch,
            q_estimator.optimizer.param_groups[0]['lr']),
                guard=args.show_loss)
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

        q_estimator.save_weights(args.out_weights, args)
        if epoch % args.score_every == 0 and epoch < args.num_epochs:
            training_samples = assign_scores(args,
                                             q_estimator,
                                             predictor,
                                             replay_memory,
                                             progress=True)
            rescore_lr *= args.rescore_gamma
            q_estimator.optimizer.param_groups[0]['lr'] = rescore_lr
	

        pass

    pass


def main():
    parser = \
        argparse.ArgumentParser()

    parser.add_argument("tmp_file")
    parser.add_argument("out_weights", type=Path2)
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
    parser.add_argument("--epoch-step", default=16, type=int)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--rescore-gamma", default=0.5, type=float)
    parser.add_argument("--show-loss", action='store_true')
    parser.add_argument("--print-every", dest="print_every",
                        type=int, default=5)

    parser.add_argument("--max-term-length", default=512, type=int)
    parser.add_argument("--num-predictions", default=16, type=int)
    parser.add_argument("--max-tuples", default=None, type=int)
    parser.add_argument("--time-discount", default=0.9, type=float)
    parser.add_argument("--score-every", default=1, type=int)

    parser.add_argument("--verbose", "-v", action='count', default=0)

    args = parser.parse_args()
    supervised_q(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import sys
import math
from typing import List

import torch

from predict_tactic import loadPredictorByFile
from dataloader import get_premise_features, tokenize, encode_fpa_stem

def main(arg_list: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("weightsfile")
    parser.add_argument("goal")
    parser.add_argument("premise")
    args = parser.parse_args()

    predictor = loadPredictorByFile(args.weightsfile)
    tokenized_goal = tokenize(predictor.dataloader_args,
                                  predictor.metadata,
                                  args.goal)
    tokenized_premise = tokenize(predictor.dataloader_args,
                                     predictor.metadata,
                                     args.premise)
    premise_features = get_premise_features(predictor.dataloader_args,
                                            predictor.metadata,
                                            args.goal,
                                            args.premise)

    stem_idxs = torch.LongTensor([encode_fpa_stem(predictor.dataloader_args,
                                                  predictor.metadata,
                                                  stem) for stem in
                                  ["apply", "rewrite"]])
    scores = predictor.hyp_name_scores(stem_idxs, tokenized_goal,
                                       [tokenized_premise], [premise_features])

    for score in scores.view(-1):
        print(math.exp(score))

    
if __name__ == "__main__":
    main(sys.argv[1:])

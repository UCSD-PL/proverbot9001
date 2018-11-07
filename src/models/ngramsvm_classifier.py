import argparse
import time
import math
import pickle
import sys

from typing import Dict, Any, List, Tuple, Iterable, cast, Union

# Using sklearn for the actual learning
from sklearn import svm

# Using Torch to get nllloss
import torch
from torch import nn
from torch.autograd import Variable

from models.tactic_predictor import TacticPredictor
from tokenizer import tokenizers
from data import get_text_data, encode_ngram_classify_data, encode_ngram_classify_input
from util import *
from serapi_instance import get_stem


class NGramSVMClassifier(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        with open(filename, 'rb') as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        assert checkpoint['stem-embeddings']
        assert checkpoint['tokenizer']
        assert checkpoint['n']
        assert checkpoint['classifier']
        assert checkpoint['options']

        self.embedding = checkpoint['stem-embeddings']
        self.tokenizer = checkpoint['tokenizer']
        self.classifier = checkpoint['classifier']
        self.n = checkpoint['n']
        self.options = checkpoint['options']

    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

    def __init__(self, options : Dict[str, Any]) -> None:
        assert options["filename"]
        self.load_saved_state(options["filename"])
        self.criterion = nn.NLLLoss()

    def predictDistribution(self, in_data : Dict[str, Union[str, List[str]]]) \
        -> torch.FloatTensor:
        goal = cast(str, in_data["goal"])
        feature_vector = encode_ngram_classify_input(goal, self.n, self.tokenizer)
        distribution = self.classifier.predict_log_proba([feature_vector])[0]
        return distribution

    def predictKTactics(self, in_data : Dict[str, Union[str, List[str]]], k : int) \
        -> List[Tuple[str, float]]:
        distribution = self.predictDistribution(in_data)
        indices, probabilities = list_topk(list(distribution), k)
        return [(self.embedding.decode_token(idx) + ".",
                 math.exp(certainty))
                for certainty, idx in zip(probabilities, indices)]

    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[str, List[str]]], k : int,
                                correct : str) -> Tuple[List[Tuple[str, float]], float]:
        distribution = self.predictDistribution(in_data)
        correct_stem = get_stem(correct)
        if self.embedding.has_token(correct_stem):
            loss = self.criterion(torch.FloatTensor(distribution).view(1, -1), Variable(torch.LongTensor([self.embedding.encode_token(correct_stem)]))).item()
        else:
            loss = float("+inf")
        indices, probabilities = list_topk(list(distribution), k)
        predictions = [(self.embedding.decode_token(idx) + ".",
                        math.exp(certainty))
                       for certainty, idx in zip(probabilities, indices)]
        return predictions, loss

Checkpoint = Tuple[Dict[Any, Any], float]

def main(args_list : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "A second-tier predictor which predicts tactic "
                                     "stems based on word frequency in the goal")
    parser.add_argument("--context-filter", dest="context_filter",
                        type=str, default="default")
    parser.add_argument("--num-keywords", dest="num_keywords",
                        type=int, default=100)
    parser.add_argument("--max-tuples", dest="max_tuples",
                        type=int, default=None)
    parser.add_argument("--gram-size", "-n", dest="n", type=int, default=1)
    parser.add_argument("scrape_file")
    parser.add_argument("save_file")
    args = parser.parse_args(args_list)
    dataset = get_text_data(args.scrape_file, args.context_filter,
                            max_tuples=args.max_tuples, verbose=True)
    samples, tokenizer, embedding = encode_ngram_classify_data(dataset, args.n,
                                                               tokenizers["no-fallback"],
                                                               args.num_keywords, 2)

    classifier, loss = train(samples, embedding.num_tokens())

    state = {'stem-embeddings': embedding,
             'tokenizer':tokenizer,
             'classifier': classifier,
             'n': args.n,
             'options': [
                 ("dataset size", str(len(samples))),
                 ("context filter", args.context_filter),
                 ("training loss", loss),
                 ("# stems", embedding.num_tokens()),
                 ("# tokens", args.num_keywords),
             ]}
    with open(args.save_file, 'wb') as f:
        pickle.dump(state, f)

def train(dataset, num_stems: int) -> Checkpoint:
    curtime = time.time()
    print("Training SVM...", end="")
    sys.stdout.flush()

    inputs, outputs = zip(*dataset)
    model = svm.SVC(gamma='scale', probability=True)
    model.fit(inputs, outputs)
    print(" {:.2f}s".format(time.time() - curtime))
    loss = model.score(inputs, outputs)
    print("Training loss: {}".format(loss))
    return model, loss

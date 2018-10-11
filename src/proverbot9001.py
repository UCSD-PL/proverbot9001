#!/usr/bin/env python3

import signal
import sys
import models.encdecrnn_predictor as encdec
import models.try_common_predictor as trycommon
import models.wordbagclass_predictor as wordbagclass
import models.encclass_predictor as encclass
import models.dnnclass_predictor as dnnclass
import models.k_nearest_predictor as knn
import models.encstruct_predictor as encstruct
import models.term_autoencoder as autoencoder
import models.autoclass_predictor as autoclass
from tokenizer import tokenizers
import report
import argparse
import data
import itertools

from typing import Dict, Callable, List

def exit_early(signal, frame):
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, exit_early)
    parser = argparse.ArgumentParser(description=
                                     "Proverbot9001 toplevel. Used for training "
                                     "and making reports")
    parser.add_argument("command", choices=list(modules.keys()))
    args = parser.parse_args(sys.argv[1:2])
    modules[args.command](sys.argv[2:])

def train(args):
    parser = argparse.ArgumentParser(description=
                                     "Proverbot9001 training module")
    parser.add_argument("model", choices=list(trainable_models.keys()))
    args_values = parser.parse_args(args[:1])
    trainable_models[args_values.model](args[1:])

def get_data(args : List[str]) -> None:
    parser = argparse.ArgumentParser(description=
                                     "Parse datafiles into multiple formats")
    parser.add_argument("format", choices=["terms"])
    parser.add_argument("datafile_path", type=str)
    parser.add_argument("--tokenizer",
                        choices=list(tokenizers.keys()), type=str,
                        default=list(tokenizers.keys())[0])
    parser.add_argument("--max-tuples", dest="max_tuples", default=None, type=int)
    parser.add_argument("--num-keywords", dest="num_keywords", default=100, type=int)
    parser.add_argument("--max-length", dest="max_length", default=None, type=int)
    parser.add_argument("--lineend", dest="lineend", default=False, const=True,
                        action='store_const')
    arg_values = parser.parse_args(args)
    if arg_values.format == "terms":
        terms, tokenizer = data.term_data(data.read_text_data(arg_values.datafile_path,
                                                              arg_values.max_tuples),
                                          tokenizers[arg_values.tokenizer],
                                          arg_values.num_keywords, 2)
        if arg_values.max_length:
            terms = [data.normalizeSentenceLength(term, arg_values.max_length)
                     for term in terms]
        for term in terms:
            print(tokenizer.toString(
                list(itertools.takewhile(lambda x: x != data.EOS_token, term))),
                  end="\\n\n" if arg_values.lineend else "\n")
        pass

modules = {
    "train" : train,
    "report":  report.main,
    "data": get_data,
}

trainable_models : Dict[str, Callable[[List[str]], None]] = {
    "encclass" : encclass.main,
    "encdec" : encdec.main,
    "encstruct" : encstruct.main,
    "dnnclass" : dnnclass.main,
    "trycommon" : trycommon.train,
    "wordbagclass" : wordbagclass.main,
    "k-nearest" : knn.main,
    "autoencoder" : autoencoder.main,
    "autoclass" : autoclass.main,
}


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import signal
import sys
import models.encdecrnn_predictor as encdec
import models.try_common_predictor as trycommon
import models.wordbagclass_predictor as wordbagclass
import models.encclass_predictor as encclass
import report
import argparse

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

modules = {
    "train" : train,
    "report":  report.main,
}

trainable_models = {
    "encdec" : encdec.main,
    "encclass" : encclass.main,
    "trycommon" : trycommon.train,
    "wordbagclass" : wordbagclass.main,
}


if __name__ == "__main__":
    main()

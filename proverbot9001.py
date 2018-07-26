#!/usr/bin/env python3

import signal
import sys
import models.encdecrnn_predictor as encdec
import models.try_common_predictor as trycommon
import report
import argparse

modules = {
    "train-encdec" : encdec.main,
    "train-trycommon" : trycommon.train,
    "report":  report.main,
}

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

if __name__ == "__main__":
    main()

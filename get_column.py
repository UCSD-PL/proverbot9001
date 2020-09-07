#!/usr/bin/env python3

import csv
import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Like cut, but respects quotes")
    parser.add_argument("-f", "--field", default=0, type=int)
    parser.add_argument("-d", "--delimiter", default=",", type=str)
    args = parser.parse_args()

    reader = csv.reader(sys.stdin, delimiter=args.delimiter)
    for row in reader:
        print(row[args.field])


if __name__ == "__main__":
    main()

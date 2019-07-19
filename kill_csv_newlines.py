#!/usr/bin/env python3

import csv
import sys
import re

def main() -> None:
    reader = csv.reader(sys.stdin)
    writer = csv.writer(sys.stdout)
    for row in reader:
        writer.writerow([re.sub("\n", "\\\\n", entry) for entry in row])

if __name__ == "__main__":
    main()

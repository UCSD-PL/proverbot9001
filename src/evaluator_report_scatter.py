##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import argparse
import json
import matplotlib.pyplot as plt
from pathlib_revised import Path2

def main():
    parser = argparse.ArgumentParser(
        description="Make a scatter plot of predicted distances vs actual distances")
    parser.add_argument("outfile", type=Path2,
                        help="Place to put the resulting graph")
    parser.add_argument("eval_json_files", nargs="+", type=Path2,
                        help="Json files from the evaluator report to use")
    parser.add_argument("--max-x", type=int)
    parser.add_argument("--max-y", type=int)
    args = parser.parse_args()

    data = []
    for json_file in args.eval_json_files:
        with json_file.open('r') as f:
            for row in f.readlines():
                try:
                    data.append(json.loads(row))
                except:
                    print(f"Couldn't load \"{row}\"")
                    raise

    data_filtered = [pnt for pnt in data
                     if (((not args.max_x) or pnt["actual-distance"] <= args.max_x)
                         and
                         ((not args.max_y) or pnt["predicted-distance"] <= args.max_y))]
    x = [pnt["actual-distance"] for pnt in data_filtered]
    y = [pnt["predicted-distance"] for pnt in data_filtered]

    plt.scatter(x, y, c='#1f77b420')
    plt.xlabel("Actual Distance")
    plt.ylabel("Predicted Distance")
    plt.savefig(args.outfile)

    pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3.7
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
import shutil
import glob
import re
import subprocess
from pathlib_revised import Path2

def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Script which checks the solutions produced by our search reports")
    parser.add_argument("--prelude", default=".", type=str,
                        help=
                        "The `home` directory in which to look for the _CoqProject file.")
    parser.add_argument("--skip-incomplete", dest="skip_incomplete",
                        help="Skip checking files that aren't finished running yet",
                        action='store_const', const=True, default=False)
    parser.add_argument("--print-stdout", dest="print_stdout",
                        help="Print the stdout of each checked file",
                        action='store_const', const=True, default=False)
    parser.add_argument("--verbose", "-v",
                        action='store_const', const=True, default=False)
    parser.add_argument("report_dir",
                        help="output data folder name",
                        default="search-report",
                        type=Path2)
    args = parser.parse_args()
    vfiles = args.report_dir.glob('*.v')

    try:
        with open(args.prelude + "/_CoqProject", 'r') as includesfile:
            includes = re.sub("\n", " ", includesfile.read())
    except FileNotFoundError:
        print("Didn't find a _CoqProject file in prelude dir. Did you forget to pass --prelude?")
        includes = ""

    for vfile in vfiles:
        check_vfile(vfile, includes, args)

def unescape_filename(filename : str) -> str:
    return re.sub("Zs", "/", re.sub("Zd", ".", re.sub("ZZ", "Z", filename)))

def check_vfile(vfile_path : Path2, includes : str, args : argparse.Namespace) -> None:
    html_path = vfile_path.with_suffix(".html")
    src_path = Path2(unescape_filename(vfile_path.stem))
    if args.skip_incomplete:
        if not html_path.exists():
            print(f"Skipping {src_path}")
            return
    else:
        assert html_path.exists(), \
            f"Couldn't find HTML file for {src_path}. "\
            f"Are you sure the report is completed?"

    src_f = src_path.with_suffix("")
    src_ext = src_path.suffix
    new_filename_path = Path2(str(src_f) + "_solution" + src_ext)
    vfile_path.copyfile(args.prelude / new_filename_path)

    result = subprocess.run(["coqc"] + includes.split() + [str(new_filename_path)],
                            cwd=args.prelude, capture_output=True,
                            encoding='utf8')
    assert result.returncode == 0, \
        f"Returned a non zero errorcode {result.returncode}! \n"\
        f"{result.stderr}"
    print(f"Checked {src_path}")
    if args.print_stdout:
        print(f"Output:\n{result.stdout}", end="")

if __name__ == "__main__":
    main()

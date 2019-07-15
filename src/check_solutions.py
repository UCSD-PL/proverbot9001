#!/usr/bin/env python3.7

import argparse
import shutil
import glob
import re
import subprocess
from os import path

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
                        default="search-report")
    args = parser.parse_args()
    vfiles = glob.glob(f"{args.report_dir}/*.v")

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

def check_vfile(vfile : str, includes : str, args : argparse.Namespace) -> None:
    html_file = path.splitext(vfile)[0] + ".html"
    src_filename = unescape_filename(path.splitext(path.basename(vfile))[0])
    if args.skip_incomplete:
        if not path.exists(html_file):
            print(f"Skipping {src_filename}")
            return
    else:
        assert path.exists(path.splitext(vfile)[0] + ".html"), \
            f"Couldn't find HTML file for {src_filename}. "\
            f"Are you sure the report is completed?"

    src_f, src_ext = path.splitext(src_filename)
    new_filename = src_f + "_solution" + src_ext

    shutil.copy(vfile, args.prelude + "/" + new_filename)

    result = subprocess.run(["coqc"] + includes.split() + [new_filename],
                            cwd=args.prelude, capture_output=True,
                            encoding='utf8')
    assert result.returncode == 0, \
        f"Returned a non zero errorcode {result.returncode}! \n"\
        f"{result.stderr}"
    print(f"Checked {src_filename}")
    if args.print_stdout:
        print(f"Output:\n{result.stdout}", end="")

if __name__ == "__main__":
    main()

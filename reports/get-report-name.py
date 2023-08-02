#!/usr/bin/env python3
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
import subprocess
import re
from glob import glob
from subprocess import check_output
from datetime import datetime

def get_line(command):
    return (check_output(command, shell=True).decode('utf-8').strip())

def get_report_name(report_dir):
    with open(report_dir + "/index.html") as f:
        contents = f.read()
    datestring_match = re.search(r"Run on (\d+-\d+-\d+ \d+:\d+:\d+.\d+)", contents)
    if not datestring_match:
        filename = glob(report_dir + "*/index.html")[0]
        with open(filename) as f:
            contents = f.read()
            datestring_match = re.search(r"Run on (\d+-\d+-\d+ \d+:\d+:\d+.\d+)", contents)
            assert datestring_match

    datestring = datestring_match.group(1)
    date = datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S.%f")
    assert date != ""
    short_commit = re.search(r"Commit: ([0-9a-f]+)", contents).group(1)
    assert short_commit != ""
    commit = get_line("git show {} | head -n 1 | cut -d ' ' -f2".format(short_commit))
    return date.strftime("%Y-%m-%dT%Hd%Md%S-0700") + "+" + commit

parser = argparse.ArgumentParser(description="produce a canonical report label from it's date and commit")
parser.add_argument("reportdir", nargs=1, help="The directory containing the report")
args = parser.parse_args()
print(get_report_name(args.reportdir[0]))

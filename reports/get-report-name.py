#!/usr/bin/env python3

import argparse
import subprocess
import re
from subprocess import check_output
from datetime import datetime

def get_line(command):
    return (check_output(command, shell=True).decode('utf-8').strip())

def get_report_name(report_dir):
    with open(report_dir + "/report.html") as f:
        contents = f.read()
        datestring = re.search(r"Run on (\d+-\d+-\d+ \d+:\d+:\d+.\d+ \w+)", contents).group(1)
        zonestring = re.search(r"Run on \d+-\d+-\d+ \d+:\d+:\d+.\d+ (\w+)", contents).group(1)
        date = datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S.%f %Z")
        assert(zonestring == "PST")
        assert(date != "")
        short_commit = re.search(r"Commit: ([0-9a-f]+)", contents).group(1)
        assert(short_commit != "")
        commit = get_line("git show {} | head -n 1 | cut -d ' ' -f2".format(short_commit))
        return date.strftime("%Y-%m-%dT%Hd%Md%S-0700") + "+" + commit

parser = argparse.ArgumentParser(description="produce a canonical report label from it's date and commit")
parser.add_argument("reportdir", nargs=1, help="The directory containing the report")
args = parser.parse_args()
print(get_report_name(args.reportdir[0]))

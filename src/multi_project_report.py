#!/usr/bin/env python3

import argparse
import subprocess
import re
import sys
import os
from typing import List, Tuple

from yattag import Doc
from pathlib_revised import Path2

from util import stringified_percent

extra_files = ["multi.css", "logo.png"]

def main(arg_list: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    args = parser.parse_args()

    multi_project_index(args.dir)

def multi_project_index(report_dir: str) -> None:

    project_reports = get_lines(f"find {report_dir}/ -maxdepth 1 -mindepth 1 -type d -not -name '.*', -not -name 'output'")

    total_theorems = 0
    total_success = 0
    project_stats: List[Tuple[str, int, int]] = []
    for project_report in project_reports:
        num_theorems, num_success = get_stats(project_report)
        total_theorems += num_theorems
        total_success += num_success
        project_stats.append((os.path.basename(project_report), num_theorems, num_success))

    write_html(report_dir, total_theorems, total_success, project_stats)
    base = Path2(os.path.abspath(__file__)).parent.parent / "reports"
    for filename in extra_files:
        (base / filename).copyfile(report_dir / filename)

def get_stats(project_dir: str) -> Tuple[int, int]:
    with open(os.path.join(project_dir, "index.html")) as f:
        contents = f.read()
    statsMatch = re.search("Proofs Completed:\s+(\d+\.\d+)%\s+\((\d+)/(\d+)\)", contents)
    assert statsMatch

    percent = statsMatch.group(1)
    num_theorems = int(statsMatch.group(3))
    num_success = int(statsMatch.group(2))
    return num_theorems, num_success

def write_html(output_dir: str, total_theorems: int, total_success:int,
               project_stats: List[Tuple[str, int, int]]):
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        with tag('head'):
            doc.stag("link", rel="stylesheet", href="multi.css")
            with tag('title'):
                text("Proverbot9001 Multi-project report")
        with tag('body'):
            with tag('img',
                     ('src', 'logo.png'),
                     ('id', 'logo')):
                pass
            with tag('h2'):
                text("Proofs completed: {}% ({}/{})"
                     .format(stringified_percent(total_success, total_theorems),
                             total_success, total_theorems))
            with tag('table'):
                with tag('tr', klass="header"):
                    line('th', "Project")
                    line('th', "% Success")
                    line('th', "# Theorems Proved")
                    line('th', "# Theorems Total")
                    line('th', "details")

                    for proj_name, num_theorems, num_success in project_stats:
                        with tag('tr'):
                            line('td', proj_name)
                            line('td', stringified_percent(num_success, num_theorems))
                            line('td', str(num_success))
                            line('td', str(num_theorems))
                            with tag('td'):
                                with tag('a', href=os.path.join(proj_name,"index.html")):
                                    text("link")
    with open(os.path.join(output_dir, "index.html"), 'w') as fout:
        fout.write(doc.getvalue())

def get_lines(command : str):
    return (subprocess.run([command], stdout=subprocess.PIPE, shell=True)
            .stdout.decode('utf-8').split('\n')[:-1])

if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3

import argparse
import subprocess
import re
from util import stringified_percent
from yattag import Doc

def main(arg_list: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    args = parser.parse_args()

    project_reports = get_lines("find -maxdepth 1 -type d -not -name '.*'")

    total_theorems = 0
    total_success = 0
    project_stats: List[Tuple[str, int, int]] = []
    for project_report in project_reports:
        num_theorems, num_success = get_stats(project_report)
        total_theorems += num_theorems
        total_success += num_success
        project_stats.append((project_report, num_theorems, num_success))

    write_html(args, total_theorems, total_success, project_stats)

def get_stats(project_dir: str):
    with open(os.path.join(project_dir, "index.html")) as f:
        contents = f.read()
    statsMatch = re.search("Proofs Completed:\s+(\d+\.\d+)%\s+\((\d+)/(\d+)\)")
    assert statsMatch

    percent = statsMatch.group(1)
    num_theorems = statsMatch.group(3)
    num_success = statsMatch.group(2)
    return num_theorems, num_success

def write_html(args: argparse.Namespace, total_theorems: int, total_success:int,
               project_stats: List[Tuple[str, int, int]]):
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        with tag('head'):
            doc.stag("link", rel="stylesheet", href="multi.css")
            with tag('title'):
                text("Proverbot9001 Multi-project report")
        with tag('body'):
            with tag('h2'):
                text("Proofs completed: {}% ({}/{})"
                     .format(stringified_percent(total_successful, total_theorems),
                             total_successful, total_theorems))
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
    with open(os.path.join(args.dir, "index.html")) as fout:
        fout.write(doc.getvalue())

def get_lines(command : str):
    return (subprocess.run([command], stdout=subprocess.PIPE)
            .stdout.decode('utf-8').split('\n')[:-1])

if __name__ == "__main__":
    main(sys.argv[1:])

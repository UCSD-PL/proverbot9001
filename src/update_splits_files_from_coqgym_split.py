
import argparse
import sys
import os.path
import json
from pathlib import Path
from glob import glob

from typing import List

def main(arg_list: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("splits_file", type=Path)
    parser.add_argument("coqgym_split_file", type=Path)
    args = parser.parse_args()

    with args.splits_file.open('r') as splits_f:
        project_dicts = json.loads(splits_f.read())

    with args.coqgym_split_file.open('r') as csplit_f:
        project_split = json.loads(csplit_f.read())

    basedir = Path(os.getcwd()) / args.root
    os.chdir(basedir)
    for project_dict in project_dicts:
        os.chdir(basedir / project_dict["project_name"])
        files = [str(Path(filename).with_suffix(".v")) for filename in
                 glob("**/*.vo", recursive=True)]
        files_filtered = []
        for filename in files:
            if not os.path.exists(filename):
                print(f"Warning: in project {project_dict['project_name']} file {str(Path(filename).with_suffix('.vo'))} doesn't have a cooresponding v file! skipping..", file=sys.stderr)
            else:
                files_filtered.append(filename)
        files = files_filtered

        if len(files) == 0:
            print(f"Warning: Project {project_dict['project_name']} has no files",
                  file=sys.stderr)
        if project_dict["project_name"] in (project_split["projs_train"]
                                            + project_split["projs_valid"]):
            project_dict["train_files"] = files
        else:
            assert project_dict["project_name"] in project_split["projs_test"], \
              f"Can't find project {project_dict['project_name']} in coqgym split!"
            project_dict["test_files"] = files

    print(json.dumps(project_dicts, indent=4))

if __name__ == "__main__":
    main(sys.argv[1:])

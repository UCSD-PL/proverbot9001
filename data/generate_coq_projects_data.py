import json
import os
import glob
from pathlib import Path

PROJECTS_SPLIT='projects-split.json'
PREFIX='../coq-projects/'

projects = json.load(open(PROJECTS_SPLIT))
train, test, valid = projects["train"], projects["test"], projects["valid"]

os.chdir(PREFIX)

def write_coq_files(dataset, txt_file):
    for d in dataset:
        for f in Path(d).glob('**/*.v'):
            txt_file.write('./' + str(f) + "\n")

for phase in ["train", "test", "valid"]:
    dataset = projects[phase]
    f = open('../data/coq-projects-{}-files.txt'.format(phase), 'w')
    write_coq_files(dataset, f)
    f.close()

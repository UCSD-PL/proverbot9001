#!/usr/bin/env python3

import subprocess
import re
from yattag import Doc
from subprocess import run
from datetime import datetime

def get_lines(command : str):
    return (run([command], shell=True, stdout=subprocess.PIPE)
            .stdout.decode('utf-8').split('\n')[:-1])

def get_file_date(filename : str) -> datetime:
    return datetime.strptime(filename.split("/")[1].split("+")[0], '%Y-%m-%dT%Hd%Md%S%z')

def get_file_percent(filename):
    with open(filename+"/report.html") as f:
        contents = f.read()
        overallPercentString = re.search(r"Overall Accuracy:\s+(\d+\.\d+)%", contents)
        if overallPercentString:
            return float(overallPercentString.group(1))
        else:
            return float(re.search(r"Searched:\s+(\d+\.\d+)%", contents).group(1))


files = sorted(get_lines("find -type d -not -name '.*'"), key=lambda f: get_file_date(f), reverse=True)

doc, tag, text, line = Doc().ttl()
with tag('html'):
    with tag('head'):
        with tag('script', src='index.js'):
            pass
        with tag('script', src='https://d3js.org/d3.v4.min.js'):
            pass
        doc.stag("link", rel="stylesheet", href="index.css")
        with tag('title'):
            text("Proverbot9001 Reports")
        pass
    with tag('body'):
        with tag('svg', width='700', height='400',
                 style="border-style:outset; border-color:#00ffff"):
            pass
    with tag('table'):
        with tag('tr', klass="header"):
            line('th', 'Date')
            line('th', 'Time')
            line('th', 'Overall Accuracy')
            line('th', '')
        for f in files:
            date = get_file_date(f)
            with tag('tr'):
                line('td', date.strftime("%a %b %d %Y"))
                line('td', date.strftime("%H:%M"))
                line('td', str(get_file_percent(f)) + "%")
                with tag('td'):
                    line('a', 'link', href=(f + "/report.html"))
with open('index.html', 'w') as index_file:
    index_file.write(doc.getvalue())

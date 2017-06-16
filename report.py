#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import math
import threading
import queue
import re

from shutil import *
from format import format_context
from yattag import Doc

import serapi_instance
import linearize_semicolons

from serapi_instance import count_fg_goals
from helper import *

num_jobs = 0
jobs = queue.Queue()
workers = []
num_tactics = 0
num_correct = 0
output_lock = threading.Lock()
finished_queue = queue.Queue()
rows = queue.Queue()
base = os.path.dirname(os.path.abspath(__file__))
darknet_command = ["{}/try-auto.py".format(base)]
vernacular_binder = [
    "Definition",
    "Inductive",
    "Fixpoint",
    "Theorem",
    "Lemma",
    "Example",
    "Ltac",
    "Record",
    "Variable",
    "Section",
    "End",
    "Instance",
    "Module",
    "Context"
]
vernacular_words = vernacular_binder + [
    # "Proof",
    "Qed",
    "Defined",
    "Require",
    "Import",
    "Print",
    "Assumptions",

]

local_binder = [
    "forall",
    "fun"
]

syntax_words = local_binder + [
    "Type",
    "Set",
    "Prop",
    "if",
    "then",
    "else",
    "match",
    "with",
    "end",
    "as",
    "in",
    "return",
    "using",
    "let"
]

vernacular_color = "a020f0"
syntax_color = "228b22"
global_bound_color = "3b10ff"
local_bound_color = "a0522d"

def color_word(color, word):
    return "<span style=\"color:{}\">{}</span>".format(color, word)

# def highlight_comments(page):
#     comment_depth = 0
#     for c in page:


def syntax_highlight(page):
    for vernac in vernacular_words:
        page = re.sub(vernac,
                      color_word(vernacular_color, vernac),
                      page)
    # for stx in syntax_words:
    #     page = re.sub("(^|\s|\(){}($|\s)".format(stx),
    #                   color_word(syntax_color, stx),
    #                   page)
    return page

def load_commands_preserve(filename):
    with open(filename, 'r') as fin:
        contents = fin.read()
    result = []
    cur_command = ""
    comment_depth = 0
    in_quote = False
    for i in range(len(contents)):
        cur_command += contents[i]
        if in_quote:
            if contents[i] == '"' and contents[i-1] != '\\':
                in_quote = False
        else:
            if contents[i] == '"' and contents[i-1] != '\\':
                in_quote = True
            elif comment_depth == 0:
                if (re.match("[\{\}]", contents[i]) and
                      re.fullmatch("\s*", cur_command[:-1])):
                    result.append(cur_command)
                    cur_command = ""
                elif (re.fullmatch("\s*[\+\-\*]+",
                                   serapi_instance.kill_comments(cur_command)) and
                      (len(contents)==i+1 or contents[i] != contents[i+1])):
                    result.append(cur_command)
                    cur_command = ""
                elif (re.match("\.($|\s)", contents[i:i+2]) and
                      (not contents[i-1] == "." or contents[i-2] == ".")):
                    result.append(cur_command)
                    cur_command = ""
            if contents[i:i+2] == '(*':
                comment_depth += 1
            elif contents[i-1:i+1] == '*)':
                comment_depth -= 1
    return result

class Worker(threading.Thread):
    def __init__(self, workerid, coqargs, includes, output_dir):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = coqargs
        self.includes = includes
        self.workerid = workerid
        self.output_dir = output_dir
        pass

    def process_file(self, filename):
        global num_tactics
        global num_correct
        num_tactics_in_file = 0
        num_correct_in_file = 0
        current_context = 0
        scripts = ""

        commands = lift_and_linearize(load_commands_preserve(filename),
                                      coqargs, includes, filename)

        doc, tag, text, line = Doc().ttl()

        with tag('html'):
            with tag('head'):
                doc.stag('link', href='details.css', rel='stylesheet')
                doc.stag('link', href='jquery-ui.css', rel='stylesheet')
                with tag('script', type='text/javascript',
                         src="http://code.jquery.com/jquery-latest.min.js"):
                    pass
                with tag('script', type='text/javascript',
                         src="jquery-ui.js"):
                    pass
                with tag('title'):
                    text("Proverbot Detailed Report for {}".format(filename))
            with serapi_instance.SerapiContext(self.coqargs, self.includes) as coq:
                with tag('body'):
                    with tag('pre'):
                        for command in commands:
                            if re.match(";", command) and options["no-semis"]:
                                coq.run_stmt(command)
                                return
                            in_proof = coq.proof_context
                            if in_proof:
                                num_tactics_in_file += 1
                                query = format_context(coq.prev_tactics, coq.get_goals(),
                                                       coq.get_hypothesis())
                                response, errors = subprocess.Popen(darknet_command,
                                                                    stdin=
                                                                    subprocess.PIPE,
                                                                    stdout=
                                                                    subprocess.PIPE,
                                                                    stderr=
                                                                    subprocess.PIPE
                                ).communicate(input=query.encode('utf-8'))
                                result = response.decode('utf-8').strip()

                                scripts += ("<script type='text/javascript'>\n"
                                            "$(function () {\n"
                                            "    $(\"#context-"
                                            + str(current_context)
                                            + "\").tooltip({\n"
                                            "        content: \"<pre><code>")
                                scripts += (coq.proof_context
                                            .replace("\\", "\\\\")
                                            .replace("\n", "\\n")
                                            .replace("    ", "  "))
                                scripts += "\n Predicted: {}\n".format(result)
                                scripts += ("</code></pre>\",\n"
                                            "        open: function (event, ui) {\n"
                                            "            ui.tooltip.css('max-width', 'none');\n"
                                            "            ui.tooltip.css('min-width', '800px');\n"
                                            "        }\n"
                                            "    });\n"
                                            "});\n"
                                            "</script>")
                                with tag('span', title='tooltip',
                                         id='context-' + str(current_context)):
                                    if command.strip() == result:
                                        num_correct_in_file += 1
                                        with tag('code', klass="goodcommand"):
                                                text(command)
                                    else:
                                        with tag('code', klass="badcommand"):
                                            text(command)
                                    current_context += 1
                            else:
                                with tag('code', klass="plaincommand"):
                                    text(command)

                            try:
                                coq.run_stmt(command)
                            except:
                                print("In file {}:".format(filename))
                                raise

        details_filename = "{}.html".format(escape_filename(filename))
        with open("{}/{}".format(self.output_dir, details_filename), "w") as fout:
            fout.write(syntax_highlight(doc.getvalue()) + scripts)

        output_lock.acquire()
        num_tactics += num_tactics_in_file
        num_correct += num_correct_in_file
        output_lock.release()
        if num_tactics_in_file > 0:
            percent_correct = (num_correct_in_file / num_tactics_in_file) * 100
        else:
            percent_correct = 0
        rows.put({'filename': filename, 'num_tactics': num_tactics_in_file,
                  'num_correct': num_correct_in_file,
                  '% correct': percent_correct,
                  'details_filename': details_filename})
    def run(self):
        try:
            while(True):
                job = jobs.get_nowait()
                print("Processing file {} ({} of {})".format(job,
                                                             num_jobs - jobs.qsize(),
                                                             num_jobs))
                self.process_file(job)
        except queue.Empty:
            pass
        finally:
            finished_queue.put(self.workerid)

def escape_filename(filename):
    return re.sub("/", "Zs", re.sub("\.", "Zd", re.sub("Z", "ZZ", filename)))

parser = argparse.ArgumentParser(description=
                                 "try to match the file by predicting a tacti")
parser.add_argument('-j', '--threads', default=1, type=int)
parser.add_argument('--prelude', default=".")
parser.add_argument('-o', '--output', help="output data folder name",
                    default="report")
parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
args = parser.parse_args()

coqargs = ["{}/coq-serapi/sertop.native".format(base),
           "--prelude={}/coq".format(base)]
includes = subprocess.Popen(['make', '-C', args.prelude, 'print-includes'],
                            stdout=subprocess.PIPE).communicate()[0].decode('utf-8')

if not os.path.exists(args.output):
    os.makedirs(args.output)

num_jobs = len(args.filenames)
for infname in args.filenames:
    jobs.put(infname)

for idx in range(args.threads):
    worker = Worker(idx, coqargs, includes, args.output)
    worker.start()
    workers.append(worker)

for idx in range(args.threads):
    finished_id = finished_queue.get()
    workers[finished_id].join()

doc, tag, text, line = Doc().ttl()

with tag('html'):
    with tag('head'):
        doc.stag('link', href='report.css', rel='stylesheet')
        with tag('title'):
            text("Proverbot Report")
    with tag('body'):
        with tag('h2'):
            if (num_tactics == 0):
                stringified_percent = "NaN"
            else:
                stringified_percent = "{:10.2f}".format((num_correct / num_tactics) * 100)
            text("Overall Accuracy: {}% ({}/{})"
                 .format(stringified_percent,
                         num_correct, num_tactics))
        with tag('table'):
            with tag('tr'):
                line('th', 'Filename')
                line('th', 'Number of Tactics in File')
                line('th', 'Number of Tactics Correctly Predicted')
                line('th', '% Correct')
                line('th', 'Details')
            while not rows.empty():
                row = rows.get()
                with tag('tr'):
                    line('td', row['filename'])
                    line('td', row['num_tactics'])
                    line('td', row['num_correct'])
                    line('td', "{:10.2f}%".format(row['% correct']))
                    with tag('td'):
                        with tag('a', href=row['details_filename']):
                            text("Details")

copy("{}/report.css".format(base), "{}/report.css".format(args.output))
copy("{}/details.css".format(base), "{}/details.css".format(args.output))
copy("{}/jquery-ui.css".format(base), "{}/jquery-ui.css".format(args.output))
copy("{}/jquery-ui.js".format(base), "{}/jquery-ui.js".format(args.output))

with open("{}/report.html".format(args.output), "w") as fout:
    fout.write(doc.getvalue())

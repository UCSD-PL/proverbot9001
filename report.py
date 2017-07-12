#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import math
import threading
import queue
import re
import datetime

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
output_lock = threading.Lock()
finished_queue = queue.Queue()
rows = queue.Queue()
base = os.path.dirname(os.path.abspath(__file__))
darknet_command = ""
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
    "Proof",
    "Qed",
    "Defined",
    "Require",
    "Import",
    "Export",
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

vernacular_color = "#a020f0"
syntax_color = "#228b22"
global_bound_color = "#3b10ff"
local_bound_color = "#a0522d"
comment_color = "#004800"

def color_word(color, word):
    return "<span style=\"color:{}\">{}</span>".format(color, word)

def highlight_comments(page):
    result = ""
    comment_depth = 0
    for i in range(len(page)):
        if(page[i:i+2] == "(*"):
            comment_depth += 1
            if comment_depth == 1:
                result += "<span style=\"color:{}\">".format(comment_color)
        result += page[i]
        if(page[i-1:i+1] == "*)"):
            comment_depth -= 1
            if comment_depth == 0:
                result += "</span>"
    return result;

def syntax_highlight(page):
    for vernac in vernacular_words:
        page = re.sub(vernac,
                      color_word(vernacular_color, vernac),
                      page)
    return highlight_comments(page);

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

css = ["details.css", "jquery-ui.css"]
javascript = ["http://code.jquery.com/jquery-latest.min.js", "jquery-ui.js"]

def details_header(tag, doc, text):
    with tag('head'):
        for filename in css:
            doc.stag('link', href=filename, rel='stylesheet')
        for filename in javascript:
            with tag('script', type='text/javascript',
                     src=filename):
                pass
        with tag('title'):
            text("Proverbot Detailed Report for {}".format(filename))

def jsan(string):
    return (string
            .replace("\\", "\\\\")
            .replace("\n", "\\n"))

def shorten_whitespace(string):
    return string.replace("    ", "  ")

def hover_script(context_idx, proof_context, predicted_tactic):
    return ("<script type='text/javascript'>\n"
            "$(function () {{\n"
            "  $(\"#context-{}\").tooltip({{\n"
            "    content: \"<pre><code>{}\\n\\n"
            "               <b>Predicted</b>: {}\\n"
            "               </code></pre>\",\n"
            "    open: function (event, ui) {{\n"
            "      ui.tooltip.css('max-width', 'none');\n"
            "      ui.tooltip.css('min-width', '800px');\n"
            "    }}\n"
            "  }});\n"
            "}});\n"
            "</script>".format(str(context_idx),
                               shorten_whitespace(jsan(proof_context)),
                               shorten_whitespace(jsan(predicted_tactic))))

num_tactics = 0
num_correct = 0
num_partial = 0
num_failed = 0

class Worker(threading.Thread):
    def __init__(self, workerid, coqargs, includes, output_dir, prelude="."):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = coqargs
        self.includes = includes
        self.workerid = workerid
        self.output_dir = output_dir
        self.prelude = prelude
        pass

    def process_file(self, filename):
        global num_tactics
        global num_correct
        global num_partial
        global num_failed
        num_tactics_in_file = 0
        num_correct_in_file = 0
        num_partial_in_file = 0
        num_failed_in_file = 0
        current_context = 0
        scripts = ""

        commands = lift_and_linearize(load_commands_preserve(self.prelude + "/" +
                                                             filename),
                                      self.coqargs, self.includes, self.prelude, filename)

        doc, tag, text, line = Doc().ttl()

        with tag('html'):
            details_header(tag, doc, text)
            with serapi_instance.SerapiContext(self.coqargs,
                                               self.includes,
                                               self.prelude) as coq:
                with tag('body'), tag('pre'):
                    for command in commands:
                        if re.match(";", command) and options["no-semis"]:
                            coq.run_stmt(command)
                            return
                        in_proof = (coq.proof_context and
                                    not re.match(".*Proof.*", command.strip()))
                        if in_proof:
                            num_tactics_in_file += 1
                            query = format_context(coq.prev_tactics, coq.get_hypothesis(),
                                                   coq.get_goals())
                            response, errors = subprocess.Popen(darknet_command,
                                                                stdin=
                                                                subprocess.PIPE,
                                                                stdout=
                                                                subprocess.PIPE,
                                                                stderr=
                                                                subprocess.PIPE
                            ).communicate(input=query.encode('utf-8'))
                            result = response.decode('utf-8', 'ignore').strip()

                            scripts += hover_script(current_context,
                                                    coq.proof_context,
                                                    result)
                            try:
                                coq.run_stmt(result)
                                failed = False
                            except:
                                failed = True
                                num_failed_in_file += 1
                            coq.cancel_last()

                            with tag('span', title='tooltip',
                                     id='context-' + str(current_context)):
                                if command.strip() == result:
                                    num_correct_in_file += 1
                                    num_partial_in_file += 1
                                    with tag('code', klass="goodcommand"):
                                        text(command)
                                elif (command.strip().split(" ")[0].strip(".") ==
                                      result.strip().split(" ")[0].strip(".")):
                                    num_partial_in_file += 1
                                    with tag('code', klass="okaycommand"):
                                        text(command)
                                elif failed:
                                    with tag('code', klass="failedcommand"):
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
        num_partial += num_partial_in_file
        num_failed += num_failed_in_file
        output_lock.release()
        if num_tactics_in_file > 0:
            percent_correct = (num_correct_in_file / num_tactics_in_file) * 100
            percent_partial = (num_partial_in_file / num_tactics_in_file) * 100
        else:
            percent_correct = 0
            percent_partial = 0
        rows.put({'filename': filename, 'num_tactics': num_tactics_in_file,
                  'num_correct': num_correct_in_file, 'num_partial': num_partial_in_file,
                  'num_failed': num_failed,
                  '% correct': percent_correct,
                  '% partial': percent_partial,
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
parser.add_argument('-p', '--predictor',
                    help="The command to use to predict tactics. This command must "
                    "accept input on standard in in the format specified in format.py, "
                    "and produce a tactic on standard out. The first \"{}\" in the "
                    "command will be replaced with the base directory.",
                    default="{}/try-auto.py")
parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
args = parser.parse_args()
darknet_command = [args.predictor.format(base)]

coqargs = ["{}/coq-serapi/sertop.native".format(base),
           "--prelude={}/coq".format(base)]
includes = subprocess.Popen(['make', '-C', args.prelude, 'print-includes'],
                            stdout=subprocess.PIPE).communicate()[0].decode('utf-8')

cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                     shell=True).decode('utf-8')
cur_date = datetime.datetime.now()

if not os.path.exists(args.output):
    os.makedirs(args.output)

num_jobs = len(args.filenames)
for infname in args.filenames:
    jobs.put(infname)

for idx in range(args.threads):
    worker = Worker(idx, coqargs, includes, args.output, args.prelude)
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
        with tag('h4'):
            text("Using predictor: {}".format(darknet_command[0]))
        with tag('h4'):
            text("{} files processed".format(num_jobs))
        with tag('h5'):
            text("Commit: {}".format(cur_commit))
        with tag('h5'):
            text("Run on {}".format(cur_date))
        with tag('table'):
            with tag('tr'):
                line('th', 'Filename')
                line('th', 'Number of Tactics in File')
                line('th', 'Number of Tactics Correctly Predicted')
                line('th', 'Number of Tactics Predicted Partially Correct')
                line('th', '% Correct')
                line('th', '% Partial')
                line('th', 'Details')
            while not rows.empty():
                row = rows.get()
                with tag('tr'):
                    line('td', row['filename'])
                    line('td', row['num_tactics'])
                    line('td', row['num_correct'])
                    line('td', row['num_partial'])
                    line('td', "{:10.2f}%".format(row['% correct']))
                    line('td', "{:10.2f}%".format(row['% partial']))
                    with tag('td'):
                        with tag('a', href=row['details_filename']):
                            text("Details")

copy("{}/report.css".format(base), "{}/report.css".format(args.output))
copy("{}/details.css".format(base), "{}/details.css".format(args.output))
copy("{}/jquery-ui.css".format(base), "{}/jquery-ui.css".format(args.output))
copy("{}/jquery-ui.js".format(base), "{}/jquery-ui.js".format(args.output))

with open("{}/report.html".format(args.output), "w") as fout:
    fout.write(doc.getvalue())

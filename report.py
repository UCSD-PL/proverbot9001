#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import threading
import queue
import re
import datetime

from shutil import *
from format import format_context
from yattag import Doc

import serapi_instance
import linearize_semicolons
from serapi_instance import ParseError, LexError

from helper import *
from syntax import syntax_highlight
from helper import load_commands_preserve

from darknet.python.darknet import load_net
from predict_tactic import *

finished_queue = queue.Queue()
rows = queue.Queue()
base = os.path.dirname(os.path.abspath(__file__))

details_css = ["details.css"]
details_javascript = ["details.js"]

num_predictions = 3

def header(tag, doc, text, css, javascript, title):
    with tag('head'):
        for filename in css:
            doc.stag('link', href=filename, rel='stylesheet')
        for filename in javascript:
            with tag('script', type='text/javascript',
                     src=filename):
                pass
        with tag('title'):
            text(title)

def details_header(tag, doc, text, filename):
    header(tag, doc, text, details_css, details_javascript,
           "Proverbot Detailed Report for {}".format(filename))

def report_header(tag, doc, text):
    header(tag, doc, text, ["report.css"], [],
           "Proverbot Report")

def stringified_percent(total, outof):
    if outof == 0:
        return "NaN"
    else:
        return "{:10.2f}".format(total * 100 / outof)

def to_list_string(l):
    return "% ".join([str(item) for item in l])

def shorten_whitespace(string):
    return re.sub("    +", "  ", string)

def run_prediction(coq, prediction_tuple):
    prediction, probability = prediction_tuple
    prediction = prediction.lstrip("-+*")
    if not "." in prediction:
        return (prediction, "", ParseError("No period"))
    else:
        coq.quiet = True
        try:
            coq.run_stmt(prediction)
            context = coq.proof_context
            coq.cancel_last()
            return (prediction, probability, context, None)
        except (ParseError, LexError, CoqExn, BadResponse) as e:
            return (prediction, probability, "", e)
        finally:
            coq.quiet = False

# Warning: Mutates fresult
def evaluate_prediction(fresult, correct_command,
                        correct_result_context, prediction_run):
    prediction, probability, context, exception = prediction_run
    grade = fresult.grade_command_result(prediction, context, correct_command,
                                         correct_result_context, exception)
    return (prediction, probability, grade)

class GlobalResult:
    def __init__(self):
        self.num_tactics = 0
        self.num_correct = 0
        self.num_partial = 0
        self.num_failed = 0
        self.num_topN = 0
        self.lock = threading.Lock()
        pass
    def add_file_result(self, result):
        self.lock.acquire()
        self.num_tactics += result.num_tactics
        self.num_correct += result.num_correct
        self.num_partial += result.num_partial
        self.num_failed += result.num_failed
        self.num_topN += result.num_topN
        self.lock.release()
        pass
    def report_results(self, doc, text, tag, line):
        with tag('h2'):
            text("Overall Accuracy: {}% ({}/{})"
                 .format(stringified_percent(self.num_correct, self.num_tactics),
                         self.num_correct, self.num_tactics))
        with tag('h3'):
            text("Top {} Accuracy: {}% ({}/{})"
                 .format(num_predictions,
                         stringified_percent(self.num_topN, self.num_tactics),
                         self.num_topN, self.num_tactics))
        with tag('table'):
            with tag('tr'):
                line('th', 'Filename')
                line('th', 'Number of Tactics in File')
                line('th', 'Number of Tactics Correctly Predicted')
                line('th', 'Number of Tactics Predicted Partially Correct')
                line('th', '% Correct')
                line('th', '% Top {}'.format(num_predictions))
                line('th', '% Partial')
                line('th', 'Details')
            while not rows.empty():
                fresult = rows.get()
                with tag('tr'):
                    line('td', fresult.filename)
                    line('td', fresult.num_tactics)
                    line('td', fresult.num_correct)
                    line('td', fresult.num_partial)
                    line('td', stringified_percent(fresult.num_correct,
                                                   fresult.num_tactics))
                    line('td', stringified_percent(fresult.num_topN,
                                                   fresult.num_tactics))
                    line('td', stringified_percent(fresult.num_partial,
                                                   fresult.num_tactics))
                    with tag('td'):
                        with tag('a', href=fresult.details_filename()):
                            text("Details")
    pass

def add_to_freq_table(table, entry):
    if entry not in table:
        table[entry] = 1
    else:
        table[entry] += 1

def get_stem(command):
    return command.strip().split(" ")[0].strip(".")

class FileResult:
    def __init__(self, filename):
        self.num_tactics = 0
        self.num_correct = 0
        self.num_partial = 0
        self.num_topN = 0
        self.num_failed = 0
        self.filename = filename
        self.actual_tactic_frequency = {}
        self.predicted_tactic_frequency = {}
        self.correctly_predicted_frequency = {}
        pass
    def grade_command_result(self, predicted, predicted_context,
                             actual, actual_context, exceptions):
        if actual.strip() == predicted.strip():
            add_to_freq_table(self.correctly_predicted_frequency,
                              get_stem(predicted))
            return "goodcommand"
        elif predicted_context == actual_context:
            add_to_freq_table(self.correctly_predicted_frequency,
                              get_stem(predicted))
            return "mostlygoodcommand"
        elif (get_stem(actual) == get_stem(predicted)):
            return "okaycommand"
        elif exceptions[0] == None:
            return "badcommand"
        elif type(exceptions[0]) == ParseError or type(exceptions[0]) == LexError:
            return "superfailedcommand"
        else:
            return "failedcommand"
    def add_command_result(self, predictions, prediction_contexts,
                           actual, actual_context, exception):
        add_to_freq_table(self.actual_tactic_frequency,
                          get_stem(actual))
        add_to_freq_table(self.predicted_tactic_frequency,
                          get_stem(predictions[0]))

        self.num_tactics += 1
        if (actual.strip() == predictions[0].strip() or
            actual_context == prediction_context[0]):
            add_to_freq_table(self.correctly_predicted_frequency,
                              get_stem(predictions[0]))
            self.num_correct += 1
            self.num_partial += 1
        elif (get_stem(actual) == get_stem(predictions[0])):
            self.num_partial += 1
        elif exception != None:
            self.num_failed += 1

        for prediction, prediction_context in zip(predictions, prediction_contexts):
            if (actual.strip() == prediction.strip() or
                actual_context == prediction_context):
                self.num_topN += 1
        pass
    def details_filename(self):
        return "{}.html".format(escape_filename(self.filename))
    pass

gresult = GlobalResult()

class Worker(threading.Thread):
    def __init__(self, workerid, coqargs, includes,
                 output_dir, prelude, debug, num_jobs):
        threading.Thread.__init__(self, daemon=True)
        self.coqargs = coqargs
        self.includes = includes
        self.workerid = workerid
        self.output_dir = output_dir
        self.prelude = prelude
        self.debug = debug
        self.num_jobs = num_jobs
        self.net = load_net("coq.test.cfg".encode('utf-8'),
                            "darknet/backup/coq.backup".encode('utf-8'), 0)
        pass

    def get_commands(self, filename):
        return lift_and_linearize(load_commands_preserve(self.prelude + "/" + filename),
                                  self.coqargs, self.includes, self.prelude,
                                  filename, debug=self.debug)

    def process_file(self, filename):
        global gresult
        fresult = FileResult(filename)
        current_context = 0
        scripts = ""

        if self.debug:
            print("Preprocessing...")
        commands = self.get_commands(filename)

        doc, tag, text, line = Doc().ttl()

        command_results = []

        with serapi_instance.SerapiContext(self.coqargs,
                                           self.includes,
                                           self.prelude) as coq:
            coq.debug = self.debug
            for command in commands:
                in_proof = (coq.proof_context and
                            not re.match(".*Proof.*", command.strip()))
                if in_proof:
                    query = format_context(coq.prev_tactics, coq.get_hypothesis(),
                                           coq.get_goals())
                    predictions = predict_tactics(self.net, query, 3)

                    hyps = coq.get_hypothesis()
                    goals = coq.get_goals()

                    prediction_runs = [run_prediction(coq, prediction) for
                                       prediction in predictions]

                    try:
                        coq.run_stmt(command)
                        actual_result_context = coq.proof_context
                    except (AckError, CompletedError, CoqExn,
                            BadResponse, ParseError, LexError):
                        print("In file {}:".format(filename))
                        raise

                    prediction_results = [evaluate_prediction(fresult, command,
                                                              actual_result_context,
                                                              prediction_run)
                                          for prediction_run in prediction_runs]
                    fresult.add_command_result(
                        [pred for pred, prob, ctxt, ex in prediction_runs],
                        [ctxt for pred, prob, ctxt, ex in prediction_runs],
                        command, actual_result_context,
                        [ex for pred, prob, ctxt, ex in prediction_runs])

                    command_results.append((command, hyps, goals,
                                            prediction_results))
                else:
                    try:
                        coq.run_stmt(command)
                    except (AckError, CompletedError, CoqExn,
                            BadResponse, ParseError, LexError):
                        print("In file {}:".format(filename))
                        raise
                    command_results.append((command,))

        with tag('html'):
            details_header(tag, doc, text, filename)
            with tag('div', id='overlay', onclick='event.stopPropagation();'):
                with tag('div', id='predicted'):
                    pass
                with tag('div', id='context'):
                    pass
                with tag('div', id='stats'):
                    pass
                pass
            with tag('body', onclick='deselectTactic()',
                     onload='setSelectedIdx()'), tag('pre'):
                for idx, command_result in enumerate(command_results):
                    if len(command_result) == 1:
                        with tag('code', klass='plaincommand'):
                            text(command_result[0])
                    else:
                        command, hyps, goal, prediction_results = command_result
                        predictions = [prediction for prediction, probability, grade in
                                       prediction_results]
                        probabilities = [probability for prediction, probability, grade in
                                         prediction_results]
                        grades = [grade for prediction, probability, grade in
                                  prediction_results]
                        with tag('span',
                                 ('data-hyps',hyps),
                                 ('data-goal',shorten_whitespace(goal)),
                                 ('data-num-total', str(fresult.num_tactics)),
                                 ('data-predictions',
                                  to_list_string(predictions)),
                                 ('data-num-predicteds',
                                  to_list_string([fresult.predicted_tactic_frequency
                                                  .get(get_stem(prediction), 0)
                                                  for prediction in predictions])),
                                 ('data-num-corrects',
                                  to_list_string([fresult.correctly_predicted_frequency
                                                  .get(get_stem(prediction), 0)
                                                  for prediction in predictions])),
                                 ('data-probabilities',
                                  to_list_string(probabilities)),
                                 ('data-grades',
                                  to_list_string(grades)),
                                 id='command-' + str(idx),
                                 onmouseover='hoverTactic({})'.format(idx),
                                 onmouseout='unhoverTactic()',
                                 onclick='selectTactic({}); event.stopPropagation();'
                                 .format(idx)):
                            with tag('code', klass=grades[0]):
                                text(command)

        with open("{}/{}".format(self.output_dir, fresult.details_filename()), "w") as fout:
            fout.write(syntax_highlight(doc.getvalue()) + scripts)

        gresult.add_file_result(fresult)
        rows.put(fresult)
    def run(self):
        try:
            while(True):
                job = jobs.get_nowait()
                jobnum = num_jobs - jobs.qsize()
                print("Processing file {} ({} of {})".format(job,
                                                             jobnum,
                                                             num_jobs))
                self.process_file(job)
                print("Finished file {} ({} of {})".format(job,
                                                           jobnum,
                                                           num_jobs))
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
parser.add_argument('--debug', default=False, const=True, action='store_const')
parser.add_argument('-o', '--output', help="output data folder name",
                    default="report")
parser.add_argument('filenames', nargs="+", help="proof file name (*.v)")
args = parser.parse_args()

coqargs = ["{}/coq-serapi/sertop.native".format(base),
           "--prelude={}/coq".format(base)]
includes = subprocess.Popen(['make', '-C', args.prelude, 'print-includes'],
                            stdout=subprocess.PIPE).communicate()[0].decode('utf-8')

# Get some metadata
cur_commit = subprocess.check_output(["git show --oneline | head -n 1"],
                                     shell=True).decode('utf-8')
cur_date = datetime.datetime.now()

if not os.path.exists(args.output):
    os.makedirs(args.output)

jobs = queue.Queue()
workers = []
num_jobs = len(args.filenames)
for infname in args.filenames:
    jobs.put(infname)

args.threads = min(args.threads, len(args.filenames))

for idx in range(args.threads):
    worker = Worker(idx, coqargs, includes, args.output,
                    args.prelude, args.debug, num_jobs)
    worker.start()
    workers.append(worker)

for idx in range(args.threads):
    finished_id = finished_queue.get()
    workers[finished_id].join()
    print("Thread {} finished ({} of {}).".format(finished_id, idx + 1, args.threads))

###
### Write the report page out
###

doc, tag, text, line = Doc().ttl()

with tag('html'):
    report_header(tag, doc, text)
    with tag('body'):
        with tag('h4'):
            text("{} files processed".format(num_jobs))
        with tag('h5'):
            text("Commit: {}".format(cur_commit))
        with tag('h5'):
            text("Run on {}".format(cur_date.strftime("%Y-%m-%d %H:%M:%S %Z"))
        result.report_results(doc, text, tag, line)

extra_files = ["report.css", "details.css", "details.js"]

for filename in extra_files:
    copy(base + "/reports/" + filename, args.output + "/" + filename)

with open("{}/report.html".format(args.output), "w") as fout:
    fout.write(doc.getvalue())

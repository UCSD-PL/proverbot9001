#!/usr/bin/env python3.7

import subprocess
import threading
import re
import queue
import os
import os.path
import argparse
import sys
import tempfile
# This dependency is in pip, the python package manager
from sexpdata import *
from traceback import *
from format import format_context, format_tactic

# We use these files to preprocess coq files and run them
from helper import *
import linearize_semicolons
import serapi_instance

from typing import Dict, Any, TextIO

# This stuff synchronizes between the different threads
num_jobs = 0
jobs = queue.Queue() # type: queue.Queue[str]
workers = []
output_lock = threading.Lock()
finished_queue = queue.Queue() # type: queue.Queue[int]

# A structure for holding global options that we don't want to muddy
# up the code by passing everywhere.
options = {} # type: Dict[str, Any]

# This is a worker thread. All the actual scraping is done by one of
# these. They basically just pull jobs from the queue and scrape
# them. They synchronize only when writing the output at the very end,
# they buffer everything until they are finished.
class Worker(threading.Thread):
    def __init__(self, output : str, workerid : int, prelude=".") -> None:
        threading.Thread.__init__(self, daemon=True)
        thispath = os.path.dirname(os.path.abspath(__file__))
        # Set up the command which runs sertop.
        self.coqargs = ["{}/../coq-serapi/sertop.native".format(thispath),
                        "--prelude={}/../coq".format(thispath)]
        # Run 'print-includes' in the prelude directory to try to get
        # any includes passed on the ocmmand line. On failure, just
        # assumes there are no includes.
        includes=subprocess.Popen(['make', '-C', prelude, 'print-includes'],
                                  stdout=subprocess.PIPE).communicate()[0]
        self.includes=includes.strip().decode('utf-8')
        # Set up some basic variables. The prelude is the base
        # directory for all paths, and the workerid is used to create
        # a temp file for each worker.
        self.prelude = prelude
        self.workerid = workerid
        self.outfile_name = output
        pass

    def process_statement(self, coq : serapi_instance.SerapiInstance, command : str,
                          result_file : TextIO) -> None:
        # When the no-semis option is enabled, skip the scraping of
        # commands with semicolons in them. We have a pass that's
        # supposed to remove these no matter what, but in some proofs
        # it's hard/impossible, so we leave those proofs be. We need
        # to leave those semis in until this step in the process,
        # because we still need to get those proof terms constructed
        # so that we can progress in the file. So, we run the
        # statement, but don't scrape it.
        if re.match(";", command) and options["no-semis"]:
            coq.run_stmt(command)
            return
        # If the context isn't None, then we are in a proof.
        in_proof = coq.proof_context
        if in_proof and not "Proof" in command:
            # Pull all the information we could possibly want to have
            # in our scrape.
            prev_tactics = coq.prev_tactics
            prev_hyps = coq.get_hypothesis()
            prev_goal = coq.get_goals()
            # rel_lemmas = coq.get_lemmas_about_head()
            # Write out all the information about the current context,
            # and the tactic that is in the file and should be run
            # next.
            result_file.write(format_context(prev_tactics, prev_hyps,
                                             prev_goal, "").encode('utf-8'))
            result_file.write(format_tactic(command).encode('utf-8'))
        # Run the actual command, advancing the coq state.
        coq.run_stmt(command)
        pass

    def process_file(self, filename : str) -> None:
        # Wrap this in a try block so that we can display the file
        # that failed if there are any weird failures.
        try:
            # Load the commands from the file, lift any theorem
            # statements or tactics within proofs, and then attempt to
            # linearize all semi-colon commands. Once this is done, we
            # have the proof we'll actually operate on. This file
            # should be semantically the same as the input file, just
            # easier to work with, so it's the right thing to learn.
            commands = try_load_lin(filename)
            if not commands:
                commands = lift_and_linearize(load_commands(filename),
                                              self.coqargs, self.includes, self.prelude,
                                              filename,
                                              options["skip-nochange-tac"],
                                              debug=options["debug"])
                save_lin(commands, filename)

            # Get a coq instance
            with serapi_instance.SerapiContext(self.coqargs,
                                               self.includes,
                                               self.prelude) as coq:
                # If we're in debug mode, let it know to print
                # things. Otherwise this is a noop.
                coq.debug = options["debug"]
                # Now, process each command.
                with tempfile.TemporaryFile() as temp_file:
                    # Scrape the file
                    for command in commands:
                        self.process_statement(coq, command, temp_file)
                    # When we're done with scraping this file, lock the output
                    # file and write the contents of our temp file to it.
                    temp_file.seek(0)
                    output_lock.acquire()
                    if self.outfile_name:
                        with open(self.outfile_name, 'a') as out_file:
                            for line in temp_file:
                                out_file.write(line.decode())
                            out_file.flush()
                    else:
                        for line in tmp_file:
                            print(line)
                    output_lock.release()
        except:
            print("In file {}:".format(filename))
            raise


    def run(self) -> None:
        # Until there are no more jobs left in the queue, pull a job,
        # let the user know you're processing it, and then process
        # that file.
        try:
            while(True):
                # The fact that this is _nowait means it will throw an
                # exception if the queue is empty, instead of just
                # blocking, which means there's nothing left for this
                # queue to do and it can safely kill itself.
                job = jobs.get_nowait()
                print("Processing file {} ({} of {})".format(job,
                                                             num_jobs - jobs.qsize(),
                                                             num_jobs))
                self.process_file(job)
        except queue.Empty:
            pass
        finally:
            # Let the parent thread know we're done.
            finished_queue.put(self.workerid)
        pass
    pass

# Parse the command line arguments.
parser = argparse.ArgumentParser(description="scrape a proof")
parser.add_argument('-o', '--output', help="output data file name", default=None)
parser.add_argument('-j', '--threads', default=1, type=int)
parser.add_argument('--prelude', default=".")
parser.add_argument('--no-semis', default=False, const=True, action='store_const',
                    dest='no_semis')
parser.add_argument('--debug', default=False, const=True, action='store_const')
parser.add_argument('--skip-nochange-tac', default=False, const=True, action='store_const',
                    dest='skip_nochange_tac')
parser.add_argument('inputs', nargs="+", help="proof file name(s) (*.v)")
args = parser.parse_args()
options["no-semis"] = args.no_semis
options["debug"] = args.debug
options["skip-nochange-tac"] = args.skip_nochange_tac

# Put each job on the work queue.
num_jobs = len(args.inputs)

for infname in args.inputs:
    jobs.put(args.prelude + "/" + infname)

# Start each thread, and keep track of it in the "workers" list.
for idx in range(args.threads):
    worker = Worker(args.output, idx, args.prelude)
    worker.start()
    workers.append(worker)

# Wait for each thread to finish.
for idx in range(args.threads):
    finished_id = finished_queue.get()
    workers[finished_id].join()

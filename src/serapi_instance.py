#!/usr/bin/env python3

import subprocess
import threading
import re
import queue
import os
import os.path
import argparse
import sys
import signal

from typing import List, Any, Optional, cast, Tuple
# This dependency is in pip, the python package manager
from sexpdata import *

from traceback import *

# Some Exceptions to throw when various responses come back from coq
class AckError(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass
class CompletedError(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass

class CoqExn(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass
class BadResponse(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass

class NotInProof(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass
class ParseError(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass

class LexError(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass
class TimeoutError(Exception):
    def __init__(self, msg : Any) -> None:
        self.msg = msg
    pass
# This is the class which represents a running Coq process with Serapi
# frontend. It runs its own thread to do the actual passing of
# characters back and forth from the process, so all communication is
# asynchronous unless otherwise noted.
class SerapiInstance(threading.Thread):
    # This takes three parameters: a string to use to run serapi, a
    # list of coq includes which the files we're running on will
    # expect, and a base directory You can also set the coq objects
    # ".debug" field after you've created it to get more verbose
    # logging.
    def __init__(self, coq_command : List[str], includes : str, prelude : str,
                 timeout : int = 30) -> None:
        # Set up some threading stuff. I'm not totally sure what
        # daemon=True does, but I think I wanted it at one time or
        # other.
        threading.Thread.__init__(self, daemon=True)
        # Open a process to coq, with streams for communicating with
        # it.
        self._proc = subprocess.Popen(coq_command,
                                      cwd=prelude,
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        self._fout = self._proc.stdout
        self._fin = self._proc.stdin
        self.timeout = timeout

        # Initialize some state that we'll use to keep track of the
        # coq state. This way we don't have to do expensive queries to
        # the other process for to answer simple questions.
        self._current_fg_goal_count = None # type: Optional[int]
        self.proof_context = None # type: Optional[str]
        self.prev_state = -1
        self.cur_state = 0
        self.prev_tactics = [] # type: List[str]


        # Set up the message queue, which we'll populate with the
        # messages from serapi.
        self.messages = queue.Queue() # type: queue.Queue[Sexp]
        # Set the debug flag to default to false.
        self.debug = False
        # Set the "extra quiet" flag (don't print on failures) to false
        self.quiet = False
        # Start the message queue thread
        self.start()
        # Go through the messages and throw away the initial feedback.
        self.discard_initial_feedback()
        # Execute the commands corresponding to include flags we were
        # passed
        self.exec_includes(includes, prelude)
        # Unset Printing Notations (to get more learnable goals?)
        self.unset_printing_notations()

    # Send some text to serapi, and flush the stream to make sure they
    # get it. NOT FOR EXTERNAL USE
    def send_flush(self, cmd : str):
        # if self.debug:
        #     print("SEND: " + cmd)
        self._fin.write(cmd.encode('utf-8'))
        self._fin.flush()
        self._current_fg_goal_count = None

    # Run a command. This is the main api function for this
    # class. Sends a single command to the running serapi
    # instance. Returns nothing: if you want a response, call one of
    # the other methods to get it.
    def run_stmt(self, stmt : str):
        if self.debug:
            print("Running statement: " + stmt.lstrip('\n')) # lstrip makes output shorter
        # We need to escape some stuff so that it doesn't get stripped
        # too early.
        stmt = stmt.replace("\\", "\\\\")
        stmt = stmt.replace("\"", "\\\"")
        # We'll wrap the actual running in a try block so that we can
        # report which command the error came from at this
        # level. Other higher level code might re-catch it.
        try:
            # Preprocess_command sometimes turns one command into two,
            # to get around some limitations of the serapi interface.
            for stm in preprocess_command(kill_comments(stmt)):
                # Send the command
                self.send_flush("(Control (StmAdd () \"{}\"))\n".format(stm))
                # Get the response, which indicates what state we put
                # serapi in.
                self.update_state()

                # Observe that state.
                self.send_flush("(Control (StmObserve {}))\n".format(self.cur_state))
                # Finally, get the result of the command
                feedbacks = self.get_feedbacks()

                # Get a new proof context, if it exists
                self.get_proof_context()

                # If we saw a new proof context, we're still in a
                # proof so append the command to our prev_tactics
                # list.
                if self.proof_context:
                    self.prev_tactics.append(stm)
                else:
                    # If we didn't see a new context, we're not in a
                    # proof anymore, so clear the prev_tactics state.
                    self.prev_tactics = []

        # If we hit a problem let the user know what file it was in,
        # and then throw it again for other handlers. NOTE: We may
        # want to make this printing togglable (at this level), since
        # sometimes errors are expected.
        except (CoqExn, BadResponse, AckError, CompletedError, TimeoutError) as e:
            self.handle_exception(e, stmt)

    def handle_exception(self, e : Exception, stmt : str):
        if not self.quiet or self.debug:
            print("Problem running statement: {}".format(stmt))
        if (type(e) == CoqExn):
            ce = cast(CoqExn, e)
            if   (type(ce.msg) == list and
                  ce.msg[0] == Symbol('CoqExn') and
                  len(ce.msg) == 4 and
                  type(ce.msg[3]) == list):
                if   (ce.msg[3][0] == Symbol('Stream.Error')):
                    self.get_completed()
                    raise ParseError("Could't parse command {}".format(stmt))
                if   (ce.msg[3][0] == 'CLexer.Error.E(3)'):
                    self.get_completed()
                    raise LexError("Couldn't lex command {}".format(stmt))
        if    (type(e) == CompletedError):
            co = cast(CompletedError, e)
            if   (type(co.msg) == list and
                  co.msg[0] == Symbol('Answer') and
                  len(co.msg) == 3 and
                  type(co.msg[2]) == list and
                  co.msg[2][0] == Symbol('CoqExn') and
                  len(co.msg[2]) == 4):
                if (type(co.msg[2][3]) == list and
                    co.msg[2][3][0] == Symbol('Stream.Error')):
                    raise ParseError("Couldn't parse command {}".format(stmt))
                elif (type(co.msg[2][3]) == list and
                      co.msg[2][3][0] == Symbol('Invalid_argument')):
                    raise ParseError("Invalid argument {}".format(stmt))
        if type(e) == TimeoutError:
            self.cancel_last()
            raise TimeoutError("Statement \"{}\" timed out.".format(stmt))

        self.cancel_last()
        raise e

    # Cancel the last command which was sucessfully parsed by
    # serapi. Even if the command failed after parsing, this will
    # still cancel it. You need to call this after a command that
    # fails after parsing, but not if it fails before.
    def cancel_last(self) -> None:
        if self.debug:
            print("Cancelling last statement from state {}".format(self.cur_state))
        # Flush any leftover messages in the queue
        while not self.messages.empty():
            self.get_message()
        # Run the cancel
        self.send_flush("(Control (StmCancel ({})))".format(self.cur_state))
        # Get the response from cancelling
        self.cur_state = self.get_cancelled()
        assert self.cur_state == self.prev_state, \
            "cur_state is {}, but prev_state was {}" \
            .format(self.cur_state, self.prev_state)
        # Go back to the previous state.
        assert self.prev_state != -1, "Can't cancel twice in a row!"
        self.prev_state = -1

    # Get the next message from the message queue, and make sure it's
    # an Ack
    def get_ack(self) -> None:
        ack = self.get_message()
        if (not isinstance(ack, list) or
            ack[0] != Symbol("Answer") or
            ack[2] != Symbol("Ack")):
            raise AckError(ack)

    # Get the next message from the message queue, and make sure it's
    # a Completed.
    def get_completed(self) -> None:
        completed = self.get_message()
        if (not isinstance(completed, list) or
            completed[0] != Symbol("Answer") or
            completed[2] != Symbol("Completed")):
            raise CompletedError(completed)

    def add_lib(self, origpath : str, logicalpath : str) -> None:
        addStm = ("(Control (StmAdd () \"Add Rec LoadPath \\\"{}\\\" as {}.\"))\n"
                  .format(origpath, logicalpath))
        self.send_flush(addStm.format(origpath, logicalpath))
        self.update_state()

    def search_about(self, symbol : str) -> List[str]:
        try:
            self.send_flush("(Control (StmAdd () \"SearchAbout {}.\"))\n".format(symbol))
            self.update_state()
            self.send_flush("(Control (StmObserve {}))\n".format(self.cur_state))
            feedbacks = self.get_feedbacks()
            return [self.ppSexpContent(lemma) for lemma in feedbacks[4:-1]]
        except (CoqExn, BadResponse, AckError, CompletedError) as e:
            self.handle_exception(e, "SearchAbout {}.".format(symbol))
        return []

    # Not adding any types here because it would require a lot of
    # casting. Will reassess when recursive types are added to mypy
    # https://github.com/python/mypy/issues/731
    def ppSexpContent(self, content):
        if content[0] == Symbol("Feedback"):
            return self.ppSexpContent(content[1][1][1][3][1][2])
        elif (content[0] == Symbol("PCData") and len(content) == 2
              and isinstance(content[1], str)):
            return content[1]
        elif (content[0] == Symbol("PCData") and len(content) == 2
              and content[1] == Symbol(".")):
            return "."
        elif (content[0] == Symbol("Element") and len(content) == 2
              and isinstance(content[1], list) and
              (content[1][0] == Symbol("constr.keyword") or
               content[1][0] == Symbol("constr.type") or
               content[1][0] == Symbol("constr.variable") or
               content[1][0] == Symbol("constr.reference") or
               content[1][0] == Symbol("constr.path"))):
            return dumps(content[1][2][0][1])
        elif isinstance(content[0], list):
            return "".join([self.ppSexpContent(item) for item in content])
        else:
            return dumps(content)

    def exec_includes(self, includes_string : str, prelude : str) -> None:
        for match in re.finditer("-R\s*(\S*)\s*(\S*)\s*", includes_string):
            self.add_lib("./" + match.group(1), match.group(2))

    def update_state(self) -> None:
        self.prev_state = self.cur_state
        self.cur_state = self.get_next_state()

    def unset_printing_notations(self) -> None:
        self.send_flush("(Control (StmAdd () \"Unset Printing Notations.\"))\n")
        self.get_next_state()

    def get_next_state(self) -> int:
        self.get_ack()

        msg = self.get_message()
        while isinstance(msg, list) and msg[0] == Symbol("Feedback"):
            msg = self.get_message()
        if (not isinstance(msg, list) or
            msg[0] != Symbol("Answer")):
            raise BadResponse(msg)
        submsg = msg[2]
        if (submsg[0] == Symbol("CoqExn")):
            raise CoqExn(submsg)
        elif submsg[0] != Symbol("StmAdded"):
            raise BadResponse(submsg)
        else:
            state_num = submsg[1]
            self.get_completed()
            return state_num

    def discard_initial_feedback(self) -> None:
        feedback1 = self.get_message()
        feedback2 = self.get_message()
        if (not isinstance(feedback1, list) or
            feedback1[0] != Symbol("Feedback") or
            not isinstance(feedback2, list) or
            feedback2[0] != Symbol("Feedback")):
            raise BadResponse("Not feedback")

    def get_message(self) -> 'Sexp':
        try:
            return self.messages.get(timeout=self.timeout)
        except queue.Empty:
            if self.debug:
                print("Command timed out! Cancelling")
            self._proc.send_signal(signal.SIGINT)
            interrupt_response = self.messages.get(timeout=self.timeout)
            assert isinstance(interrupt_response, list)
            assert interrupt_response[0] == Symbol("Feedback")
            assert len(interrupt_response) > 1, \
                "too short! interrupt_reponse: {}".format(interrupt_response)
            assert isinstance(interrupt_response[1], list), \
                "interrupt_response[1]: {}".format(interrupt_response[1])
            assert len(interrupt_response[1]) > 2
            assert isinstance(interrupt_response[1][1], list)
            assert interrupt_response[1][1][0] == Symbol("contents")
            assert isinstance(interrupt_response[1][1][1], list)
            assert interrupt_response[1][1][1][0] == Symbol("Message")
            assert interrupt_response[1][1][1][1] == Symbol("Error")

            interrupt_response2 = self.messages.get(timeout=self.timeout)
            assert isinstance(interrupt_response2, list)
            assert len(interrupt_response2) > 2
            assert interrupt_response2[0] == Symbol("Answer")
            assert interrupt_response2[2][0] == Symbol("CoqExn")
            assert interrupt_response2[2][3] == Symbol("Sys.Break")

            raise TimeoutError("")

    def get_feedbacks(self) -> List['Sexp']:
        self.get_ack()

        feedbacks = [] #type: List[Sexp]
        next_message = self.get_message()
        while(isinstance(next_message, list) and
              next_message[0] == Symbol("Feedback")):
            feedbacks.append(next_message)
            next_message = self.get_message()
        fin = next_message
        if (not isinstance(fin, list) or
            fin[0] != Symbol("Answer")):
            raise BadResponse(fin)
        if (isinstance(fin[2], list) or
            fin[2] != Symbol("Completed")):
            raise BadResponse(fin)

        return feedbacks

    def count_fg_goals(self) -> int:
        if self._current_fg_goal_count == None:
            # was:
            # self.send_flush("(Query ((pp ( (pp_format PpSer) (pp_depth 1) ))) Goals)\n")
            self.send_flush("(Control (StmQuery () \"all: let n := numgoals in idtac n.\"))")
            try:
                fb = self.get_feedbacks()
                # OMG this is horrible
                self._current_fg_goal_count = int(fb[-1][-1][-2][1][3][1][2][0][1]) # type: ignore
            except (CoqExn, BadResponse) as e:
                # print("count failure")
                self._current_fg_goal_count = 0
        # print("COUNT: {}".format(str(self._current_fg_goal_count)))
        return cast(int, self._current_fg_goal_count)

    def get_cancelled(self) -> int:
        finished = False
        while not finished:
            supposed_ack = self.get_message()
            if (not isinstance(supposed_ack, list) or
                supposed_ack[0] != Symbol("Answer")):
                raise AckError("Symbol is not an ack! {}".format(supposed_ack))
            if supposed_ack[2] == Symbol("Ack"):
                finished = True
        feedback = self.get_message()
        if (not isinstance(feedback, list) or
            feedback[0] != Symbol("Feedback")):
            raise BadResponse(feedback)
        subfeed = feedback[1]
        if (not isinstance(subfeed[0] , list)):
            raise BadResponse(subfeed)
        subsubfeed = subfeed[0][1]
        if (subsubfeed[0] != Symbol("State")):
            raise BadResponse(subsubfeed)
        cancelled = self.get_message()
        self.get_completed()
        return subsubfeed[1]

    def extract_proof_context(self, raw_proof_context : 'Sexp') -> str:
        return cast(List[List[str]], raw_proof_context)[0][1]

    def get_goals(self) -> str:
        assert isinstance(self.proof_context, str)
        if self.proof_context == "":
            return ""
        split = re.split("\n======+\n", self.proof_context)
        return split[1]

    def get_hypothesis(self) -> List[str]:
        assert isinstance(self.proof_context, str)
        if self.proof_context == "":
            return []
        return parse_hyps(re.split("\n======+\n", self.proof_context)[0])

    def get_proof_context(self) -> None:
        self.send_flush("(Query ((sid {}) (pp ((pp_format PpStr)))) Goals)".format(self.cur_state))
        self.get_ack()

        proof_context_message = self.get_message()
        if (not isinstance(proof_context_message, list) or
            proof_context_message[0] != Symbol("Answer")):
            raise BadResponse(proof_context_message)
        else:
            ol_msg = proof_context_message[2]
            if (ol_msg[0] != Symbol("ObjList")):
                raise BadResponse(proof_context_message)
            if len(ol_msg[1]) != 0:
                newcontext = self.extract_proof_context(ol_msg[1])
                self.proof_context = newcontext.split("\n\n")[0]
            else:
                self.proof_context = None

    def get_lemmas_about_head(self) -> str:
        goal_head = self.get_goals().split()[0]
        if (goal_head == "forall"):
            return ""
        try:
            return "\n".join(self.search_about(goal_head))
        except:
            return ""

    def run(self) -> None:
        while(True):
            line = self._fout.readline().decode('utf-8')
            if line == '': break
            response = loads(line)
            # print("Got message {}".format(response))
            self.messages.put(response)

    def kill(self) -> None:
        self._proc.terminate()
        self._proc.stdout.close()
        threading.Thread.join(self)

    pass

class SerapiContext:
    def __init__(self, coq_commands : List[str], includes : str, prelude : str) -> None:
        self.coq_commands = coq_commands
        self.includes = includes
        self.prelude = prelude
    def __enter__(self) -> SerapiInstance:
        self.coq = SerapiInstance(self.coq_commands, self.includes, self.prelude)
        return self.coq
    def __exit__(self, type, value, traceback):
        self.coq.kill()

def possibly_starting_proof(command : str) -> bool:
    stripped_command = kill_comments(command).strip()
    return (re.match("Lemma\s", stripped_command) != None or
            re.match("Theorem\s", stripped_command) != None or
            re.match("Remark\s", stripped_command) != None or
            re.match("Proposition\s", stripped_command) != None or
            re.match("Definition\s", stripped_command) != None or
            re.match("Example\s", stripped_command) != None or
            re.match("Fixpoint\s", stripped_command) != None or
            re.match("Corollary\s", stripped_command) != None or
            re.match("Let\s", stripped_command) != None or
            ("Instance" in stripped_command and
             "Declare" not in stripped_command) or
            re.match("Function\s", stripped_command) != None or
            re.match("Next Obligation", stripped_command) != None or
            re.match("Property\s", stripped_command) != None or
            re.match("Add Morphism\s", stripped_command) != None)

def ending_proof(command : str) -> bool:
    stripped_command = kill_comments(command).strip()
    return ("Qed" in stripped_command or
            "Defined" in stripped_command or
            (re.match("\s*Proof\s+\S+\s*", stripped_command) != None and
             re.match("\s*Proof\s+with", stripped_command) == None))

def split_commands(string : str) -> List[str]:
    result = []
    next_command = ""
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = False
        else:
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = True
            if (re.match("[\{\}]", string[i]) and
                re.fullmatch("\s*", next_command)):
                result.append(string[i])
                next_command = ""
                continue
            if (re.match("[\+\-\*]", string[i]) and
                string[i] != string[i+1] and
                re.fullmatch("\s*[\+\-\*]*", next_command)):
                next_command += string[i]
                result.append(next_command.strip())
                next_command = ""
                continue
            if (re.match("\.($|\s)", string[i:i+2]) and
                (not string[i-1] == "." or string[i-2] == ".")):
                result.append(next_command.strip() + ".")
                next_command = ""
                continue
        next_command += string[i]
    return result

def kill_comments(string: str) -> str:
    result = ""
    depth = 0
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if depth == 0:
                result += string[i]
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = False
        else:
            if string[i:i+2] == '(*':
                depth += 1
            if depth == 0:
                result += string[i]
            if string[i-1:i+1] == '*)':
                depth -= 1
            if string[i] == '"' and string[i-1] != '\\':
               in_quote = True
    return result

def preprocess_command(cmd : str) -> List[str]:
    needPrefix = ["String", "Classical", "ClassicalFacts",
                  "ClassicalDescription", "ClassicalEpsilon",
                  "Equivalence", "Init.Wf", "Program.Basics",
                  "Max", "Wf_nat", "EquivDec", "Znumtheory",
                  "Bool", "Zquot", "FSets", "FSetAVL",
                  "Wellfounded", "FSetInterface",
                  "OrderedType", "Program", "Recdef", "Eqdep_dec",
                  "FunctionalExtensionality", "Zwf", "Permutation",
                  "Orders", "Mergesort", "List", "ZArith", "Int31",
                  "Syntax", "Streams", "Equality",
                  "ProofIrrelevance", "Setoid", "EqNat",
                  "Arith", "Cyclic31", "Omega", "Relations",
                  "RelationClasses", "OrderedTypeAlt", "FMapAVL",
                  "BinPos", "BinNat", "DecidableClass", "Reals",
                  "Psatz", "ExtrOcamlBasic", "ExtrOcamlString",
                  "Ascii"]
    for lib in needPrefix:
        match = re.fullmatch("\s*Require(\s+(?:(?:Import)|(?:Export)))?((?:\s+\S+)*)\s+({})\s*((?:\s+\S*)*)\.".format(lib), cmd)
        if match:
            if match.group(1):
                impG= match.group(1)
            else:
                impG=""
            if match.group(4):
                after = match.group(4)
            else:
                after=""
            if (re.fullmatch("\s*", match.group(2)) and re.fullmatch("\s*", after)):
                return ["From Coq Require" + impG + " " + match.group(3) + "."]
            else:
                return ["From Coq Require" + impG + " " + match.group(3) + "."] + preprocess_command("Require " + impG.strip() + " " + match.group(2).strip() + " " + after + ".")
    return [cmd]

def get_stem(tactic : str) -> str:
    return split_tactic(tactic)[0]

def split_tactic(tactic : str) -> Tuple[str, str]:
    tactic = kill_comments(tactic).strip()
    if re.match("[-+*\{\}]", tactic):
        return tactic, ""
    if re.match(".*;.*", tactic):
        return tactic, ""
    for prefix in ["try", "now", "repeat"]:
        prefix_match = re.match("{}\s+(.*)".format(prefix), tactic)
        if prefix_match:
            rest_stem, rest_rest = split_tactic(prefix_match.group(1))
            return prefix + " " + rest_stem, rest_rest
    for special_stem in ["rewrite\s+<-", "intros until", "simpl in"]:
        special_match = re.match("{}\s+(.*)".format(special_stem), tactic)
        if special_match:
            return special_stem, special_match.group(1)
    match = re.match("^\(?(\w+)(?:\s+(.*))?", tactic)
    assert match, "tactic \"{}\" doesn't match!".format(tactic)
    stem, rest = match.group(1, 2)
    if not rest:
        rest = ""
    return stem, rest

def parse_hyps(hyps_str : str) -> List[str]:
    hyps_replaced = re.sub(":=.*?:", ":",
                           kill_nested("forall", ",",
                                       kill_nested("fun", "=>",
                                                   kill_nested("let\s", "\sin\s",
                                                               hyps))),
                           flags=re.DOTALL)
    var_terms = re.findall("(\S+(?:, \S+)*) (?::=.*?)?: .*?",
                           hyps_replaced, flags=re.DOTALL)
    rest_hyps_str = hyps_str
    hyps_list = []
    for next_term in var_terms[1:]:
        next_match = re.search(next_term, rest_hyps_str)
        assert next_match is not None, \
            "Can't find var term {} in hypothesis!".format(next_term)
        hyp = rest_hyps_str[:next_match.start()]
        rest_hyps_str = rest_hyps_str[next_match.start():]
        hyps_list.append(hyp)
    hyps_list.append(rest_hyps_str)
    return hyps_list

def kill_nested(start_string : str, end_string : str, hyps : str) -> str:
    next_forall_match = re.search(start_string, hyps, flags=re.DOTALL)
    next_comma_match = re.search(end_string, hyps, flags=re.DOTALL)
    forall_depth = 0
    last_forall_position = -1
    cur_position = 0
    while next_forall_match != None or next_comma_match != None:
        next_comma_position = next_comma_match.start()
        next_forall_position = next_forall_match.start()
        if next_forall_position < next_comma_position:
            cur_position = next_forall_position
            if forall_depth == 0:
                last_forall_position = next_forall_position
            forall_depth += 1
        else:
            if forall_depth == 1:
                hyps = hyps[:last_forall_position] + hyps[next_comma_position:]
                cur_position = last_forall_position
                last_forall_position = -1
            else:
                cur_position = next_comma_position
            forall_depth -= 1
        next_forall_match = \
            re.search(start_string, hyps[cur_position:], flags=re.DOTALL) + cur_position
        next_comma_match = \
            re.search(end_string, hyps[cur_position:], flags=re.DOTALL) + cur_position
    return hyps

def get_var_term_in_hyp(hyp : str) -> str:
    return hyp.split(":")[0].strip()
def get_vars_in_hyps(hyps : List[str]) -> List[str]:
    var_terms = [get_var_term_in_hyp(hyp) for hyp in hyps]
    var_names = [name.strip() for term in var_terms for name in term.split(",")]
    return var_names

def get_first_var_in_hyp(hyp : str) -> str:
    return get_var_term_in_hyp(hyp).split(",")[0].strip()

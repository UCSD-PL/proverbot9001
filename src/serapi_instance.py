#!/usr/bin/env python3.7

import subprocess
import threading
import re
import queue
import os
import os.path
import argparse
import sys
import signal
from dataclasses import dataclass

from typing import List, Any, Optional, cast, Tuple, Union
# These dependencies is in pip, the python package manager
from pampy import match, _, TAIL

from sexpdata import *
from traceback import *
from util import *
from format import ScrapedTactic
import tokenizer

# Some Exceptions to throw when various responses come back from coq
@dataclass
class AckError(Exception):
    msg : Union['Sexp', str]
@dataclass
class CompletedError(Exception):
    msg : 'Sexp'

@dataclass
class CoqExn(Exception):
    msg : 'Sexp'
@dataclass
class BadResponse(Exception):
    msg : 'Sexp'

@dataclass
class NotInProof(Exception):
    msg : str
@dataclass
class ParseError(Exception):
    msg : str

@dataclass
class LexError(Exception):
    msg : str
@dataclass
class TimeoutError(Exception):
    msg : str
@dataclass
class OverflowError(Exception):
    msg : str
@dataclass
class UnrecognizedError(Exception):
    msg : str
class NoSuchGoalError(Exception):
    pass
@dataclass
class CoqAnomaly(Exception):
    msg : str

def raise_(ex):
    raise ex

from typing import NamedTuple

class Subgoal(NamedTuple):
    hypotheses : List[str]
    goal : str

class FullContext(NamedTuple):
    subgoals : List[Subgoal]

@dataclass
class TacticTree:
    children : List[Union['TacticTree', str]]
    def __repr__(self) -> str:
        result = "["
        for child in self.children:
            result += repr(child)
            result += ","
        result += "]"
        return result

class TacticHistory:
    __tree : TacticTree
    __cur_subgoal_depth : int
    __subgoal_tree : List[List[Subgoal]]
    def __init__(self) -> None:
        self.__tree = TacticTree([])
        self.__cur_subgoal_depth = 0
        self.__subgoal_tree = []
    def openSubgoal(self, background_subgoals : List[Subgoal]) -> None:
        curTree = self.__tree
        for _ in range(self.__cur_subgoal_depth):
            assert isinstance(curTree.children[-1], TacticTree)
            curTree = curTree.children[-1]
        curTree.children.append(TacticTree([]))
        self.__cur_subgoal_depth += 1

        self.__subgoal_tree.append(background_subgoals)
        pass

    def closeSubgoal(self) -> None:
        assert self.__cur_subgoal_depth > 0
        self.__cur_subgoal_depth -= 1
        self.__subgoal_tree.pop()
        pass

    def curDepth(self) -> int:
        return self.__cur_subgoal_depth

    def addTactic(self, tactic : str) -> None:
        curTree = self.__tree
        for _ in range(self.__cur_subgoal_depth):
            assert isinstance(curTree.children[-1], TacticTree)
            curTree = curTree.children[-1]
        curTree.children.append(tactic)
        pass

    def removeLast(self, all_subgoals : List[Subgoal]) -> None:
        assert len(self.__tree.children) > 0, "Tried to remove from an empty tactic history!"
        curTree = self.__tree
        for _ in range(self.__cur_subgoal_depth):
            assert isinstance(curTree.children[-1], TacticTree)
            curTree = curTree.children[-1]
        if len(curTree.children) == 0:
            parent = self.__tree
            for _ in range(self.__cur_subgoal_depth-1):
                assert isinstance(parent.children[-1], TacticTree)
                parent = parent.children[-1]
            parent.children.pop()
            self.__cur_subgoal_depth -= 1
            self.__subgoal_tree.pop()
        else:
            lastChild = curTree.children[-1]
            if isinstance(lastChild, str):
                curTree.children.pop()
            else:
                assert isinstance(lastChild, TacticTree)
                self.__cur_subgoal_depth += 1
                self.__subgoal_tree.append(all_subgoals)
        pass

    def getCurrentHistory(self) -> List[str]:
        def generate() -> Iterable[str]:
            curTree = self.__tree
            for i in range(self.__cur_subgoal_depth+1):
                yield from (child for child in curTree.children if isinstance(child, str))
                if i < self.__cur_subgoal_depth:
                    assert isinstance(curTree.children[-1], TacticTree)
                    curTree = curTree.children[-1]
            pass
        return list(generate())

    def getFullHistory(self) -> List[str]:
        def generate(tree : TacticTree) -> Iterable[str]:
            for child in tree.children:
                if isinstance(child, TacticTree):
                    yield "{"
                    yield from generate(child)
                    yield "}"
                else:
                    yield child
        return list(generate(self.__tree))

    def getAllBackgroundSubgoals(self) -> List[Subgoal]:
        return [item for lst in self.__subgoal_tree for item in reversed(lst)]

    def getNextCancelled(self) -> str:
        curTree = self.__tree
        assert len(curTree.children) > 0, \
            "Tried to cancel from an empty history"
        for i in range(self.__cur_subgoal_depth):
            assert isinstance(curTree.children[-1], TacticTree)
            curTree = curTree.children[-1]

        if len(curTree.children) == 0:
            return "{"
        elif isinstance(curTree.children[-1], TacticTree):
            return "}"
        else:
            assert isinstance(curTree.children[-1], str), curTree.children[-1]
            return curTree.children[-1]

    def __str__(self) -> str:
        return f"depth {self.__cur_subgoal_depth}, {repr(self.__tree)}"

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
        # the other process to answer simple questions.
        self._current_fg_goal_count = None # type: Optional[int]
        self.proof_context = None # type: Optional[str]
        self.full_context = None # type: Optional[FullContext]
        self.cur_state = 0
        self.tactic_history = TacticHistory()

        # Set up the message queue, which we'll populate with the
        # messages from serapi.
        self.message_queue = queue.Queue() # type: queue.Queue[Sexp]
        # Set the debug flag to default to false.
        self.debug = False
        # Set the "extra quiet" flag (don't print on failures) to false
        self.quiet = False
        # Start the message queue thread
        self.start()
        # Go through the messages and throw away the initial feedback.
        self.discard_feedback()
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

    def send_acked(self, cmd : str):
        self.send_flush(cmd)
        self.get_ack()

    def ask(self, cmd : str):
        assert self.message_queue.empty()
        self.send_acked(cmd)
        msg = self.get_message(complete=True)
        return msg

    @property
    def messages(self):
        return [dumps(msg) for msg in list(self.message_queue.queue)]

    # Run a command. This is the main api function for this
    # class. Sends a single command to the running serapi
    # instance. Returns nothing: if you want a response, call one of
    # the other methods to get it.
    def run_stmt(self, stmt : str, timeout : Optional[int]=None):
        if timeout:
            old_timeout = self.timeout
            self.timeout = timeout
        assert self.message_queue.empty(), self.messages
        if re.match(r"\s*[{]\s*", stmt):
            self.run_stmt("Unshelve.")
        eprint("Running statement: " + stmt.lstrip('\n'),
               guard=self.debug) # lstrip makes output shorter
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
                # Get initial context
                context_before = self.full_context
                # Send the command
                assert self.message_queue.empty()
                self.send_acked("(Add () \"{}\")\n".format(stm))
                # Get the response, which indicates what state we put
                # serapi in.
                self.update_state()
                self.get_completed()
                assert self.message_queue.empty()

                # Execute the statement.
                self.send_acked("(Exec {})\n".format(self.cur_state))
                # Finally, get the result of the command
                feedbacks = self.get_feedbacks()
                # Get a new proof context, if it exists
                self.get_proof_context()

                if possibly_starting_proof(stm) and self.full_context:
                    self.tactic_history = TacticHistory()
                    self.tactic_history.addTactic(stm)
                elif re.match(r"\s*[{]\s*", stm):
                    assert context_before
                    self.tactic_history.openSubgoal(context_before.subgoals[1:])
                elif re.match(r"\s*[}]\s*", stm):
                    self.tactic_history.closeSubgoal()
                elif self.full_context:
                    # If we saw a new proof context, we're still in a
                    # proof so append the command to our prev_tactics
                    # list.
                    self.tactic_history.addTactic(stm)

        # If we hit a problem let the user know what file it was in,
        # and then throw it again for other handlers. NOTE: We may
        # want to make this printing togglable (at this level), since
        # sometimes errors are expected.
        except (CoqExn, BadResponse, AckError, CompletedError, TimeoutError) as e:
            self.handle_exception(e, stmt)
        finally:
            if timeout:
                self.timeout=old_timeout

    @property
    def prev_tactics(self):
        return self.tactic_history.getCurrentHistory()

    def handle_exception(self, e : Exception, stmt : str):
        eprint("Problem running statement: {}\n{}".format(stmt, e),
               guard=(not self.quiet or self.debug))
        match(e,
              TimeoutError, lambda *args: progn(self.tactic_history.addTactic(stmt), # type: ignore
                                                self.cancel_last(), # type: ignore
                                                raise_(TimeoutError("Statment \"{}\" timed out."
                                                                    .format(stmt)))),
              _, lambda e:
              match(normalizeMessage(e.msg),
                    ['Stream\.Error', str],
                    lambda *args: progn(self.get_completed(),
                                        raise_(ParseError("Couldn't parse command {}"
                                                          .format(stmt)))),

                    ['CLexer.Error.E(3)'],
                    lambda *args: progn(self.get_completed(),
                                        raise_(ParseError("Couldn't parse command {}"
                                                          .format(stmt)))),
                    'Not_found',
                    lambda *args: progn(self.tactic_history.addTactic(stmt), # type: ignore
                                        self.cancel_last(), raise_(e)), # type: ignore
                    ['CErrors\.UserError', _],
                    lambda inner: progn(self.tactic_history.addTactic(stmt), # type: ignore
                                        self.get_completed(),
                                        self.cancel_last(), raise_(e)), # type: ignore
                    ['ExplainErr\.EvaluatedError', TAIL],
                    lambda inner: progn(self.tactic_history.addTactic(stmt), # type: ignore
                                        self.get_completed(),
                                        self.cancel_last(), raise_(e)), # type: ignore
                    ['Proofview.NoSuchGoals(1)'],
                    lambda inner: progn(self.tactic_history.addTactic(stmt), # type: ignore
                                        self.cancel_last(), raise_(NoSuchGoalError())), # type: ignore

                    ['Answer', int, ['CoqExn', _, _, _, 'Stream\\.Error']],
                    lambda *args: raise_(ParseError("Couldn't parse command {}".format(stmt))),

                    ['Answer', int, ['CoqExn', _, _, _, 'Invalid_argument']],
                    lambda *args: raise_(ParseError("Invalid argument{}".format(stmt))),
                    ['Answer', int, ['CoqExn', _, _, ["Backtrace", []], [str]]],
                    lambda sentence, loc1, loc2, inner:
                    progn(self.get_completed(), raise_(CoqAnomaly("Overflowed")))
                    if re.search("Stack overflow", inner) else
                    progn(self.get_completed(), raise_(UnrecognizedError(inner))),
                    ['Answer', int, ['CoqExn', _, _, ["Backtrace", []], [str, str]]],
                    lambda sentence, loc1, loc2, inner1, inner2:
                    progn(self.get_completed(), raise_(CoqExn(inner1 + inner2))),
                    [str],
                    lambda contents:
                    progn(self.get_completed(), raise_(CoqExn(contents)))
                    if re.search("Anomaly", contents) else
                    raise_(UnrecognizedError(contents)),
                    _, lambda *args: progn(raise_(UnrecognizedError(args)))))

    # Flush all messages in the message queue
    def flush_queue(self) -> None:
        while not self.message_queue.empty():
            self.get_message()
    def sexpToTermStr(self, sexp) -> str:
        answer = self.ask(f"(Print ((pp_format PpStr)) (CoqConstr {dumps(sexp)}))")
        return match(normalizeMessage(answer),
                     ["Answer", int, ["ObjList", [["CoqString", _]]]],
                     lambda statenum, s: str(s),
                     ["Answer", int, ["CoqExn", list, list, list, _]],
                     lambda statenum, a, b, c, msg:
                     raise_(CoqExn(msg)))

    # Cancel the last command which was sucessfully parsed by
    # serapi. Even if the command failed after parsing, this will
    # still cancel it. You need to call this after a command that
    # fails after parsing, but not if it fails before.
    def cancel_last(self) -> None:
        cancelled = self.tactic_history.getNextCancelled()
        assert self.message_queue.empty(), self.messages
        context_before = self.full_context
        if context_before:
            old_subgoals = context_before.subgoals
            eprint(f"Cancelling {cancelled} "
                   f"from state {self.cur_state}",
                   guard=self.debug)
        else:
            old_subgoals = []
            eprint(f"Cancelling vernac "
                   f"from state {self.cur_state}")
        # Run the cancel
        self.send_acked("(Cancel ({}))".format(self.cur_state))
        # Get the response from cancelling
        self.cur_state = self.get_cancelled()
        # Get a new proof context, if it exists
        self.get_proof_context()

        # Fix up the previous tactics
        if context_before:
            self.tactic_history.removeLast(old_subgoals)
        if not self.full_context:
            self.tactic_history = TacticHistory()
        assert self.message_queue.empty(), self.messages
        if re.match(r"\s*[{]\s*", cancelled):
            self.cancel_last()

    # Get the next message from the message queue, and make sure it's
    # an Ack
    def get_ack(self) -> None:
        ack = self.get_message()
        match(normalizeMessage(ack),
              ["Answer", _, "Ack"], lambda state: None,
              _, lambda msg: raise_(AckError(dumps(ack))))

    # Get the next message from the message queue, and make sure it's
    # a Completed.
    def get_completed(self) -> Any:
        completed = self.get_message()
        match(normalizeMessage(completed),
              ["Answer", int, "Completed"], lambda state: None,
              _, lambda msg: raise_(CompletedError(completed)))

    def add_lib(self, origpath : str, logicalpath : str) -> None:
        addStm = ("(Add () \"Add Rec LoadPath \\\"{}\\\" as {}.\")\n"
                  .format(origpath, logicalpath))
        self.send_acked(addStm)
        self.update_state()
        self.get_completed()
        self.send_acked("(Exec {})\n".format(self.cur_state))
        self.discard_feedback()
        self.discard_feedback()
        self.get_completed()

    def search_about(self, symbol : str) -> List[str]:
        try:
            self.send_acked("(Add () \"SearchAbout {}.\")\n".format(symbol))
            self.update_state()
            self.get_completed()
            self.send_acked("(Exec {}))\n".format(self.cur_state))
            feedbacks = self.get_feedbacks()
            return [self.ppSexpContent(lemma) for lemma in feedbacks[4:-1]]
        except (CoqExn, BadResponse, AckError, CompletedError) as e:
            self.handle_exception(e, "SearchAbout {}.".format(symbol))
        return []

    # Not adding any types here because it would require a lot of
    # casting. Will reassess when recursive types are added to mypy
    # https://github.com/python/mypy/issues/731
    def ppSexpContent(self, content):
        if content[0] == "Feedback":
            return self.ppSexpContent(content[1][1][1][3][1][2])
        elif (content[0] == "PCData" and len(content) == 2
              and isinstance(content[1], str)):
            return content[1]
        elif (content[0] == "PCData" and len(content) == 2
              and content[1] == "."):
            return "."
        elif (content[0] == "Element" and len(content) == 2
              and isinstance(content[1], list) and
              (content[1][0] == "constr.keyword" or
               content[1][0] == "constr.type" or
               content[1][0] == "constr.variable" or
               content[1][0] == "constr.reference" or
               content[1][0] == "constr.path")):
            return dumps(content[1][2][0][1])
        elif isinstance(content[0], list):
            return "".join([self.ppSexpContent(item) for item in content])
        else:
            return dumps(content)

    def exec_includes(self, includes_string : str, prelude : str) -> None:
        for match in re.finditer("-R\s*(\S*)\s*(\S*)\s*", includes_string):
            self.add_lib("./" + match.group(1), match.group(2))

    def update_state(self) -> None:
        self.cur_state = self.get_next_state()

    def unset_printing_notations(self) -> None:
        self.send_acked("(Add () \"Unset Printing Notations.\")\n")
        self.get_next_state()
        self.get_completed()

    def get_next_state(self) -> int:
        msg = self.get_message()
        while match(normalizeMessage(msg),
                    ["Feedback", TAIL], lambda tail: True,
                    _, lambda x: False):
            msg = self.get_message()

        return match(normalizeMessage(msg),
                     ["Answer", int, list],
                     lambda state_num, contents:
                     match(contents,
                           ["CoqExn", _, _, _, list],
                           lambda loc1, loc2, loc3, inner:
                           raise_(CoqExn(inner)),
                           ["Added", int, TAIL],
                           lambda state_num, tail: state_num),
                     _, lambda x: raise_(BadResponse(msg)))
    def discard_feedback(self) -> None:
        feedback_message = self.get_message()
        while feedback_message[1][3][1] != Symbol("Processed"):
            feedback_message = self.get_message()

    def discard_initial_feedback(self) -> None:
        feedback1 = self.get_message()
        feedback2 = self.get_message()
        match(normalizeMessage(feedback1), ["Feedback", TAIL], lambda *args: None,
              _, lambda *args: raise_(BadResponse(feedback1)))
        match(normalizeMessage(feedback2), ["Feedback", TAIL], lambda *args: None,
              _, lambda *args: raise_(BadResponse(feedback2)))
    def interrupt(self) -> None:
        self._proc.send_signal(signal.SIGINT)
        self.flush_queue()

    def get_message(self, complete=False) -> Any:
        try:
            msg = self.message_queue.get(timeout=self.timeout)
            if complete:
                self.get_completed()
            return msg
        except queue.Empty:
            eprint("Command timed out! Interrupting", guard=self.debug)
            self._proc.send_signal(signal.SIGINT)
            num_breaks = 1
            try:
                interrupt_response = \
                    self.message_queue.get(timeout=self.timeout)
            except:
                self._proc.send_signal(signal.SIGINT)
                num_breaks += 1
                try:
                    interrupt_response = \
                        self.message_queue.get(timeout=self.timeout)
                except:
                    raise CoqAnomaly("Timing Out")

            got_answer_after_interrupt = match(normalizeMessage(interrupt_response),
                                               ["Answer", int, ["CoqExn", TAIL]],
                                               lambda *args: False,
                                               ["Answer", TAIL],
                                               lambda *args: True,
                                               _, False)
            if got_answer_after_interrupt:
                self.get_completed()
                for i in range(num_breaks):
                    try:
                        msg = self.message_queue.get(timeout=self.timeout)
                    except:
                        raise CoqAnomaly("Timing out")
                    assert isBreakMessage(msg), msg
                assert self.message_queue.empty()
                return interrupt_response
            elif isBreakMessage(interrupt_response):
                raise TimeoutError("")
            elif isBreakAnswer(interrupt_response):
                self.get_completed()
                raise TimeoutError("")
            elif match(normalizeMessage(interrupt_response, depth=10),
                       ["Feedback", [["doc_id", int], ["span_id", int], ["route", int],
                                     ["contents", ["Message", "Error", [],
                                                   ["Pp_box", ["Pp_hovbox", int],
                                                    ["Pp_glue", ["Pp_force_newline", ["Pp_string", "User interrupt."]]]]]]]],
                       lambda *args: True,
                       _, lambda *args: False):
                for i in range(num_breaks):
                    try:
                        msg = self.message_queue.get(timeout=self.timeout)
                    except:
                        raise CoqAnomaly("Timing out")
                    assert isBreakAnswer(msg), msg
                self.get_completed()
                assert self.message_queue.empty(), self.messages
                raise TimeoutError("")
            elif interrupt_response[0] == Symbol("Feedback"):
                eprint(interrupt_response)
                self.get_completed()
                assert self.message_queue.empty()
                return interrupt_response
            assert False, (interrupt_response, self.messages)

    def get_feedbacks(self) -> List['Sexp']:
        feedbacks = [] #type: List[Sexp]
        next_message = self.get_message()
        while(isinstance(next_message, list) and
              next_message[0] == Symbol("Feedback")):
            feedbacks.append(next_message)
            next_message = self.get_message()
        fin = next_message
        match(normalizeMessage(fin),
              ["Answer", _, "Completed", TAIL], lambda *args: None,
              ['Answer', _, ["CoqExn", _, _, _, _]],
              lambda statenum, loc1, loc2, loc3, inner: raise_(CoqExn(inner)),
              _, lambda *args: progn(eprint(f"message is \"{repr(fin)}\""), raise_(UnrecognizedError(fin)))
        )

        return feedbacks

    def count_fg_goals(self) -> int:
        if not self.full_context:
            return 0
        return len(self.full_context.subgoals)

    def get_cancelled(self) -> int:
        try:
            feedback = self.get_message()

            new_statenum = \
                match(normalizeMessage(feedback),
                      ["Answer", int, ["CoqExn", _, _, _, _]],
                      lambda *args: raise_(CoqExn(feedback)),
                      ["Feedback", [['doc_id', int], ['span_id', int], TAIL]],
                      lambda docnum, statenum, *rest: statenum,
                      _, lambda *args: raise_(BadResponse(feedback)))

            cancelled_answer = self.get_message()
            old_statenum = \
                match(normalizeMessage(cancelled_answer),
                      ["Answer", int, ["Canceled", list]],
                      lambda _, statenums: min(statenums),
                      ["Answer", int, ["CoqExn", _, _, _, _]],
                      lambda *args: raise_(CoqExn(cancelled_answer)),
                      _, lambda *args: raise_(BadResponse(cancelled_answer)))
        finally:
            self.get_completed()

        return new_statenum

    def extract_proof_context(self, raw_proof_context : 'Sexp') -> str:
        return cast(List[List[str]], raw_proof_context)[0][1]

    @property
    def goals(self) -> str:
        assert isinstance(self.proof_context, str)
        if self.proof_context == "":
            return ""
        split = re.split("\n======+\n", self.proof_context)
        return split[1]

    @property
    def hypotheses(self) -> List[str]:
        assert isinstance(self.proof_context, str)
        if self.proof_context == "":
            return []
        return parse_hyps(re.split("\n======+\n", self.proof_context)[0])

    @property
    def fullContext(self) -> FullContext:
        assert self.full_context
        fg_goals = self.full_context.subgoals
        return FullContext(fg_goals + self.tactic_history.getAllBackgroundSubgoals())

    def unshelve(self) -> None:
        self.send_acked("(Add () \"Unshelve.\")\n")
        self.update_state()
        self.get_completed()
        assert self.message_queue.empty()
        self.send_acked("(Exec {})\n".format(self.cur_state))
        self.discard_feedback()
        self.discard_feedback()
        self.get_completed()
        assert self.message_queue.empty()

    def get_proof_context(self) -> None:
        self.send_acked("(Query ((sid {}) (pp ((pp_format PpStr)))) Goals)".format(self.cur_state))

        proof_context_message = self.get_message()
        self.get_completed()
        if (not isinstance(proof_context_message, list) or
            proof_context_message[0] != Symbol("Answer")):
            raise BadResponse(proof_context_message)
        else:
            ol_msg = proof_context_message[2]
            if (ol_msg[0] != Symbol("ObjList")):
                raise BadResponse(proof_context_message)
            if len(ol_msg[1]) != 0:
                # If we're in a proof, then let's run Unshelve to get
                # the real goals. Note this would fail if we were not
                # in a proof, so we have to check that first.
                self.unshelve()

                # Now actually get the goals
                # Now actually get the real goals
                self.send_acked("(Query ((sid {}) (pp ((pp_format PpStr)))) Goals)"
                                .format(self.cur_state))
                proof_context_message = self.get_message()
                ol_msg = proof_context_message[2]
                self.get_completed()
                assert self.message_queue.empty()

                # Do some basic parsing on the context
                newcontext = self.extract_proof_context(ol_msg[1])
                self.proof_context = newcontext.split("\n\n")[0]
                if newcontext == "":
                    self.full_context = FullContext([])
                else:
                    # Try to do this the right way, fall back to the
                    # wrong way if we run into this bug:
                    # https://github.com/ejgallego/coq-serapi/issues/150
                    try:
                        response = self.ask("(Query () Goals)")
                        subgoal_sexps = response[2][1][0][1][0][1]
                        subgoals = []
                        for goal_sexp in subgoal_sexps:
                            goal_term = self.sexpToTermStr(goal_sexp[1][1])

                            hyps = []
                            for hyp_sexp in goal_sexp[2][1]:
                                ids_str = ",".join([dumps(var_sexp[1]) for var_sexp in hyp_sexp[0]])
                                hyp_type = self.sexpToTermStr(hyp_sexp[2])

                                hyps.append(f"{ids_str} : {hyp_type}")
                            subgoals.append(Subgoal(hyps, goal_term))
                        self.full_context = FullContext(subgoals)
                    except CoqExn:
                        if newcontext == "none":
                            self.full_context =  FullContext([])
                        else:
                            self.full_context = \
                                FullContext([parsePPSubgoal(substr) for substr
                                             in re.split("\n\n|(?=\snone)", newcontext)
                                             if substr.strip()])
                        pass
                # Cancel the Unshelve, to keep things clean.
                self.send_acked("(Cancel ({}))".format(self.cur_state))
                self.cur_state = self.get_cancelled()
                assert self.message_queue.empty()

            else:
                self.proof_context = None
                self.full_context = None

    def get_lemmas_about_head(self) -> str:
        goal_head = self.goals.split()[0]
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
            try:
                response = loads(line)
            except:
                eprint("Couldn't parse Sexp:\n{}".format(line))
                raise
            # print("Got message {}".format(response))
            self.message_queue.put(response)

    def kill(self) -> None:
        self._proc.terminate()
        self._proc.stdout.close()
        threading.Thread.join(self)

    pass

def isBreakMessage(msg : 'Sexp') -> bool:
    return match(normalizeMessage(msg),
                 "Sys\\.Break", lambda *args: True,
                 _, lambda *args: False)
def isBreakAnswer(msg : 'Sexp') -> bool:
    return match(normalizeMessage(msg),
                 ["Answer", int, ["CoqExn", [], list,
                                  ["Backtrace", []],
                                  "Sys\\.Break"]],
                 lambda *args: True,
                 _, lambda *args: False)

import contextlib
from typing import Iterator

@contextlib.contextmanager
def SerapiContext(coq_commands : List[str], includes : str, prelude : str) -> Iterator[Any]:
    coq = SerapiInstance(coq_commands, includes, prelude)
    yield coq
    coq.kill()

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
            "Admitted" in stripped_command or
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
            if string[i-1:i+1] == '*)' and depth > 0:
                depth -= 1
            if string[i] == '"' and string[i-1] != '\\':
               in_quote = True
    return result

def next_proof(cmds : Iterator[str]) -> Iterable[str]:
    next_cmd = next(cmds)
    assert possibly_starting_proof(next_cmd), next_cmd
    while not ending_proof(next_cmd):
        yield next_cmd
        next_cmd = next(cmds)
    yield next_cmd

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
                  "Ascii", "FunInd"]
    for lib in needPrefix:
        match = re.fullmatch("\s*Require(\s+(?:(?:Import)|(?:Export)))?((?:\s+\S+)*)\s+({})\s*((?:\s+\S*)*)\.\s*".format(lib), cmd)
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
    return [cmd] if cmd.strip() else []

def get_stem(tactic : str) -> str:
    return split_tactic(tactic)[0]

def split_tactic(tactic : str) -> Tuple[str, str]:
    tactic = kill_comments(tactic).strip()
    if re.match("[-+*\{\}]", tactic):
        stripped = tactic.strip()
        return stripped[:-1], stripped[-1]
    if re.match(".*;.*", tactic):
        return tactic, ""
    for prefix in ["try", "now", "repeat", "decide"]:
        prefix_match = re.match("{}\s+(.*)".format(prefix), tactic)
        if prefix_match:
            rest_stem, rest_rest = split_tactic(prefix_match.group(1))
            return prefix + " " + rest_stem, rest_rest
    for special_stem in ["rewrite <-", "rewrite !", "intros until", "simpl in"]:
        special_match = re.match("{}\s*(.*)".format(special_stem), tactic)
        if special_match:
            return special_stem, special_match.group(1)
    match = re.match("^\(?(\w+)(?:\s+(.*))?", tactic)
    assert match, "tactic \"{}\" doesn't match!".format(tactic)
    stem, rest = match.group(1, 2)
    if not rest:
        rest = ""
    return stem, rest

def parse_hyps(hyps_str : str) -> List[str]:
    lets_killed = kill_nested("\Wlet\s", "\sin\s", hyps_str)
    funs_killed = kill_nested("\Wfun\s", "=>", lets_killed)
    foralls_killed = kill_nested("\Wforall\s", ",", funs_killed)
    fixs_killed = kill_nested("\Wfix\s", ":=", foralls_killed)
    structs_killed = kill_nested("\W\{\|\s", "\|\}", fixs_killed)
    hyps_replaced = re.sub(":=.*?:(?!=)", ":", structs_killed, flags=re.DOTALL)
    var_terms = re.findall("(\S+(?:, \S+)*) (?::=.*?)?:(?!=)\s.*?",
                           hyps_replaced, flags=re.DOTALL)
    if len(var_terms) == 0:
        return []
    rest_hyps_str = hyps_str
    hyps_list = []
    # Assumes hypothesis are printed in reverse order, because for
    # whatever reason they seem to be.
    for next_term in reversed(var_terms[1:]):
        next_match = rest_hyps_str.rfind(" " + next_term + " :")
        hyp = rest_hyps_str[next_match:].strip()
        rest_hyps_str = rest_hyps_str[:next_match].strip()
        hyps_list.append(hyp)
    hyps_list.append(rest_hyps_str)
    for hyp in hyps_list:
        assert re.search(":(?!=)", hyp) != None, \
            "hyp: {}, hyps_str: {}\nhyps_list: {}\nvar_terms: {}"\
            .format(hyp, hyps_str, hyps_list, var_terms)
    return hyps_list

def kill_nested(start_string : str, end_string : str, hyps : str) \
    -> str:
    def searchpos(pattern : str, hyps : str, end : bool = False):
        match = re.search(pattern, hyps, flags=re.DOTALL)
        if match:
            if end:
                return match.end()
            else:
                return match.start()
        else:
            return float("Inf")
    next_forall_pos = searchpos(start_string, hyps)
    next_comma_pos = searchpos(end_string, hyps, end=True)
    forall_depth = 0
    last_forall_position = -1
    cur_position = 0
    while next_forall_pos != float("Inf") or (next_comma_pos != float("Inf") and forall_depth > 0):
        old_forall_depth = forall_depth
        if next_forall_pos < next_comma_pos:
            cur_position = next_forall_pos
            if forall_depth == 0:
                last_forall_position = next_forall_pos
            forall_depth += 1
        else:
            if forall_depth == 1:
                hyps = hyps[:last_forall_position] + hyps[next_comma_pos:]
                cur_position = last_forall_position
                last_forall_position = -1
            else:
                cur_position = next_comma_pos
            if forall_depth > 0:
                forall_depth -= 1

        new_next_forall_pos = \
            searchpos(start_string, hyps[cur_position+1:]) + cur_position + 1
        new_next_comma_pos = \
            searchpos(end_string, hyps[cur_position+1:], end=True) + cur_position + 1
        assert new_next_forall_pos != next_forall_pos or \
            new_next_comma_pos != next_comma_pos or \
                forall_depth != old_forall_depth, \
            "old start pos was {}, new start pos is {}, old end pos was {},"\
            "new end pos is {}, cur_position is {}"\
            .format(next_forall_pos, new_next_forall_pos, next_comma_pos,
                    new_next_comma_pos, cur_position)
        next_forall_pos = new_next_forall_pos
        next_comma_pos = new_next_comma_pos
    return hyps

def get_var_term_in_hyp(hyp : str) -> str:
    return hyp.partition(":")[0].strip()
def get_hyp_type(hyp : str) -> str:
    if re.search(":(?!=)", hyp) == None:
        return ""
    return re.split(":(?!=)", hyp, maxsplit=1)[1].strip()
def get_vars_in_hyps(hyps : List[str]) -> List[str]:
    var_terms = [get_var_term_in_hyp(hyp) for hyp in hyps]
    var_names = [name.strip() for term in var_terms for name in term.split(",")]
    return var_names

def get_indexed_vars_in_hyps(hyps : List[str]) -> List[Tuple[str, int]]:
    var_terms = [get_var_term_in_hyp(hyp) for hyp in hyps]
    var_names = [(name.strip(), hyp_idx)
                 for hyp_idx, term in enumerate(var_terms)
                 for name in term.split(",")]
    return var_names

def get_first_var_in_hyp(hyp : str) -> str:
    return get_var_term_in_hyp(hyp).split(",")[0].strip()

def normalizeMessage(sexp, depth : int=5):
    if depth <= 0:
        return sexp
    if isinstance(sexp, list):
        return [normalizeMessage(item, depth=depth-1) for item in sexp]
    if isinstance(sexp, Symbol):
        return dumps(sexp)
    else:
        return sexp

def tacticTakesHypArgs(stem : str) -> bool:
    now_match = re.match("\s*now\s+(.*)", stem)
    if now_match:
        return tacticTakesHypArgs(now_match.group(1))
    try_match = re.match("\s*try\s+(.*)", stem)
    if try_match:
        return tacticTakesHypArgs(try_match.group(1))
    return (
        stem == "apply"
        or stem == "eapply"
        or stem == "eexploit"
        or stem == "exploit"
        or stem == "erewrite"
        or stem == "rewrite"
        or stem == "erewrite !"
        or stem == "rewrite !"
        or stem == "erewrite <-"
        or stem == "rewrite <-"
        or stem == "destruct"
        or stem == "elim"
        or stem == "eelim"
        or stem == "inversion"
        or stem == "monadInv"
        or stem == "pattern"
        or stem == "revert"
        or stem == "exact"
        or stem == "eexact"
        or stem == "simpl in"
        or stem == "fold"
        or stem == "generalize"
        or stem == "exists"
        or stem == "case"
        or stem == "inv"
        or stem == "subst"
    )

def tacticTakesBinderArgs(stem : str) -> bool:
    return stem == "induction"

def tacticTakesIdentifierArg(stem : str) -> bool:
    return stem == "unfold"

def progn(*args):
    return args[-1]

def lemma_name_from_statement(stmt : str) -> str:
    lemma_match = re.match("\s*\S+\s+(\w+)", stmt)
    assert lemma_match, stmt
    lemma_name = lemma_match.group(1)
    assert ":" not in lemma_name, stmt
    return lemma_name

def get_binder_var(goal : str, binder_idx : int) -> Optional[str]:
    paren_depth = 0
    binders_passed = 0
    skip = False
    forall_match = re.match("forall\s+", goal.strip())
    if not forall_match:
        return None
    rest_goal = goal[forall_match.end():]
    for w in tokenizer.get_words(rest_goal):
        if w == "(":
            paren_depth += 1
        elif w == ")":
            paren_depth -= 1
            if paren_depth == 1 or paren_depth == 0:
                skip = False
        elif (paren_depth == 1 or paren_depth == 0) and not skip:
            if w == ":":
                skip = True
            else:
                binders_passed += 1
                if binders_passed == binder_idx:
                    return w
    return None

def normalizeNumericArgs(datum : ScrapedTactic) -> ScrapedTactic:
    numerical_induction_match = re.match("(induction|destruct)\s+(\d+)\.", datum.tactic.strip())
    if numerical_induction_match:
        stem = numerical_induction_match.group(1)
        binder_idx = int(numerical_induction_match.group(2))
        binder_var = get_binder_var(datum.goal, binder_idx)
        if binder_var:
            newtac = stem + " " + binder_var + "."
            return ScrapedTactic(datum.prev_tactics, datum.hypotheses,
                                 datum.goal, newtac)
        else:
            return datum
    else:
        return datum

def parsePPSubgoal(substr : str) -> Subgoal:
    split = re.split("\n====+\n", substr)
    assert len(split) == 2, substr
    hypsstr, goal = split
    return Subgoal(parse_hyps(hypsstr), goal)

def summarizeContext(context : FullContext) -> None:
    eprint("Context:")
    for i, subgoal in enumerate(context.subgoals):
        hyps_str = ",".join(get_first_var_in_hyp(hyp) for hyp in subgoal.hypotheses)
        goal_str = re.sub("\n", "\\n", subgoal.goal)[:100]
        eprint(f"S{i}: {hyps_str} -> {goal_str}")

def isValidCommand(command : str) -> bool:
    command = kill_comments(command)
    return ((command.strip()[-1] == "." and not re.match("\s*{", command)) or re.fullmatch("\s*[-+*{}]*\s*", command) != None) \
        and (command.count('(') == command.count(')'))

def load_commands(filename : str) -> List[str]:
    with open(filename, 'r') as fin:
        contents = kill_comments(fin.read())
        commands_orig = split_commands(contents)
        commands_preprocessed = [newcmd for cmd in commands_orig
                                 for newcmd in preprocess_command(cmd)]
        return commands_preprocessed

def load_commands_preserve(args : argparse.Namespace, file_idx : int,
                           filename : str) -> List[str]:
    with open(filename, 'r') as fin:
        contents = fin.read()
    return read_commands_preserve(args, file_idx, contents)

from tqdm import tqdm
from typing import Pattern, Match
def read_commands_preserve(args : argparse.Namespace, file_idx : int,
                           contents : str) -> List[str]:
    result = []
    cur_command = ""
    comment_depth = 0
    in_quote = False
    curPos = 0
    def search_pat(pat : Pattern) -> Tuple[Optional[Match], int]:
        match = pat.search(contents, curPos)
        return match, match.end() if match else len(contents) + 1
    with tqdm(total=len(contents)+1, file=sys.stdout,
              disable=(not args.progress),
              position = (file_idx * 2),
              desc="Reading file", leave=False,
              dynamic_ncols=True, bar_format=mybarfmt) as pbar:
      while curPos < len(contents):
          _, next_quote = search_pat(re.compile(r"(?<!\\)\""))
          _, next_open_comment = search_pat(re.compile(r"\(\*"))
          _, next_close_comment = search_pat(re.compile(r"\*\)"))
          _, next_bracket = search_pat(re.compile(r"[\{\}]"))
          next_bullet_match, next_bullet = search_pat(re.compile(r"[\+\-\*]+(?![\)\+\-\*])"))
          _, next_period = search_pat(re.compile(r"(?<!\.)\.($|\s)|\.\.\.($|\s)"))
          nextPos = min(next_quote,
                        next_open_comment, next_close_comment,
                        next_bracket,
                        next_bullet, next_period)
          assert curPos < nextPos
          next_chunk = contents[curPos:nextPos]
          cur_command += next_chunk
          pbar.update(nextPos - curPos)
          if nextPos == next_quote:
              if comment_depth == 0:
                  in_quote = not in_quote
          elif nextPos == next_open_comment:
              if not in_quote:
                  comment_depth += 1
          elif nextPos == next_close_comment:
              if not in_quote:
                  comment_depth -= 1
          elif nextPos == next_bracket:
              if not in_quote and comment_depth == 0 and \
                 re.match("\s*$", kill_comments(cur_command[:-1])):
                  result.append(cur_command)
                  cur_command = ""
          elif nextPos == next_bullet:
              assert next_bullet_match
              match_length = next_bullet_match.end() - next_bullet_match.start()
              if not in_quote and comment_depth == 0 and \
                 re.match("\s*$", kill_comments(cur_command[:-match_length])):
                  result.append(cur_command)
                  cur_command = ""
              assert next_bullet_match.end() >= nextPos
          elif nextPos == next_period:
              if not in_quote and comment_depth == 0:
                  result.append(cur_command)
                  cur_command = ""
          curPos = nextPos
      return result

def try_load_lin(args : argparse.Namespace, file_idx : int, filename : str) \
    -> Optional[List[str]]:
    if args.verbose:
        eprint("Attempting to load cached linearized version from {}"
               .format(filename + '.lin'))
    if not os.path.exists(filename + '.lin'):
        return None
    file_hash = hash_file(filename)
    with open(filename + '.lin', 'r') as f:
        if file_hash == f.readline().strip():
            return read_commands_preserve(args, file_idx, f.read())
        else:
            return None

def save_lin(commands : List[str], filename : str) -> None:
    output_file = filename + '.lin'
    with open(output_file, 'w') as f:
        print(hash_file(filename), file=f)
        for command in commands:
            print(command, file=f)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Module for interacting with a coq-serapi instance from Python (3).")
    parser.add_argument("--prelude", default=".", type=str,
                        help=
                        "The `home` directory in which to look for the _CoqProject file.")
    parser.add_argument("--includes", default=None, type=str,
                        help=
                        "The include options to pass to coq, as a single string. "
                        "If none are provided, we'll attempt to read a _CoqProject "
                        "located in the prelude directory, and fall back to no arguments "
                        "if none exists.")
    parser.add_argument("--sertop", default="sertop",
                        dest="sertopbin", type=str,
                        help=
                        "The location of the serapi (sertop) binary to use.")
    parser.add_argument("--srcfile", "-f", nargs='*', dest='srcfiles', default=[], type=str,
                        help=
                        "Coq source file(s) to execute.")
    parser.add_argument("--interactive", "-i",
                        action='store_const', const=True, default=False,
                        help=
                        "Drop into a pdb prompt after executing source file(s). "
                        "A `coq` object will be in scope as an instance of SerapiInstance, "
                        "and will kill the process when you leave.")
    parser.add_argument("--verbose", "-v",
                        action='store_const', const=True, default=False)
    parser.add_argument("--progress",
                        action='store_const', const=True, default=False)
    args = parser.parse_args()
    includes = ""
    if args.includes:
        includes = args.includes
    else:
        with contextlib.suppress(FileNotFoundError):
            with open(f"{args.prelude}/_CoqProject", 'r') as includesfile:
                includes = includesfile.read()
    thispath = os.path.dirname(os.path.abspath(__file__))
    with SerapiContext([args.sertopbin],
                       includes, args.prelude) as coq:
        def handle_interrupt(*args):
            nonlocal coq
            print("Running coq interrupt")
            coq.interrupt()

        with sighandler_context(signal.SIGINT, handle_interrupt):
            for srcpath in args.srcfiles:
                commands = load_commands_preserve(args, 0, f"{srcpath}")
                for cmd in commands:
                    eprint(f"Running: \"{cmd}\"")
                    coq.run_stmt(cmd)
            if args.interactive:
                breakpoint()
                x = 50

if __name__ == "__main__":
    main()

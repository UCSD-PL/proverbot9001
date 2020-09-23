#!/usr/bin/env python3.7
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import subprocess
import threading
import re
import queue
from pathlib_revised import Path2
import argparse
import sys
import signal
import functools
from dataclasses import dataclass
import contextlib

from typing import (List, Any, Optional, cast, Tuple, Union, Iterable,
                    Iterator, Pattern, Match, TYPE_CHECKING)
from tqdm import tqdm
# These dependencies is in pip, the python package manager
from pampy import match, _, TAIL

if TYPE_CHECKING:
    from sexpdata import Sexp
from sexpdata import Symbol, loads, dumps, ExpectClosingBracket
from util import (split_by_char_outside_matching, eprint, mybarfmt,
                  hash_file, sighandler_context, unwrap, progn)
from format import ScrapedTactic, TacticContext, Obligation, ProofContext
import tokenizer

from dataloader import rust_parse_sexp_one_level


# Some Exceptions to throw when various responses come back from coq
@dataclass
class SerapiException(Exception):
    msg: Union['Sexp', str]


@dataclass
class AckError(SerapiException):
    pass


@dataclass
class CompletedError(SerapiException):
    pass


@dataclass
class CoqExn(SerapiException):
    pass


@dataclass
class BadResponse(SerapiException):
    pass


@dataclass
class NotInProof(SerapiException):
    pass


@dataclass
class ParseError(SerapiException):
    pass


@dataclass
class LexError(SerapiException):
    pass


@dataclass
class TimeoutError(SerapiException):
    pass


@dataclass
class OverflowError(SerapiException):
    pass


@dataclass
class UnrecognizedError(SerapiException):
    pass


@dataclass
class NoSuchGoalError(SerapiException):
    pass


@dataclass
class CoqAnomaly(SerapiException):
    pass


def raise_(ex):
    raise ex


@dataclass
class TacticTree:
    children: List[Union['TacticTree', str]]

    def __repr__(self) -> str:
        result = "["
        for child in self.children:
            result += repr(child)
            result += ","
        result += "]"
        return result


class TacticHistory:
    __tree: TacticTree
    __cur_subgoal_depth: int
    __subgoal_tree: List[List[Obligation]]

    def __init__(self) -> None:
        self.__tree = TacticTree([])
        self.__cur_subgoal_depth = 0
        self.__subgoal_tree = []

    def openSubgoal(self, background_subgoals: List[Obligation]) -> None:
        curTree = self.__tree
        for i in range(self.__cur_subgoal_depth):
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

    def addTactic(self, tactic: str) -> None:
        curTree = self.__tree
        for i in range(self.__cur_subgoal_depth):
            assert isinstance(curTree.children[-1], TacticTree)
            curTree = curTree.children[-1]
        curTree.children.append(tactic)
        pass

    def removeLast(self, all_subgoals: List[Obligation]) -> None:
        assert len(self.__tree.children) > 0, \
            "Tried to remove from an empty tactic history!"
        curTree = self.__tree
        for i in range(self.__cur_subgoal_depth):
            assert isinstance(curTree.children[-1], TacticTree)
            curTree = curTree.children[-1]
        if len(curTree.children) == 0:
            parent = self.__tree
            for i in range(self.__cur_subgoal_depth-1):
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
                yield from (child for child in curTree.children
                            if isinstance(child, str))
                if i < self.__cur_subgoal_depth:
                    assert isinstance(curTree.children[-1], TacticTree)
                    curTree = curTree.children[-1]
            pass
        return list(generate())

    def getFullHistory(self) -> List[str]:
        def generate(tree: TacticTree) -> Iterable[str]:
            for child in tree.children:
                if isinstance(child, TacticTree):
                    yield "{"
                    yield from generate(child)
                    yield "}"
                else:
                    yield child
        return list(generate(self.__tree))

    def getAllBackgroundObligations(self) -> List[Obligation]:
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
    # expect, and a base directory
    def __init__(self, coq_command: List[str], module_name: str, prelude: str,
                 timeout: int = 30, use_hammer: bool = False,
                 log_outgoing_messages: Optional[Path2] = None) -> None:
        try:
            with open(prelude + "/_CoqProject", 'r') as includesfile:
                includes = includesfile.read()
        except FileNotFoundError:
            includes = ""
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
        self.log_outgoing_messages = log_outgoing_messages

        # Initialize some state that we'll use to keep track of the
        # coq state. This way we don't have to do expensive queries to
        # the other process to answer simple questions.
        self.proof_context = None  # type: Optional[ProofContext]
        self.cur_state = 0
        self.tactic_history = TacticHistory()
        self._local_lemmas: List[Tuple[str, bool]] = []

        # Set up the message queue, which we'll populate with the
        # messages from serapi.
        self.message_queue = queue.Queue()  # type: queue.Queue[str]
        # Set the debug flag to default to false.
        self.verbose = 0
        # Set the "extra quiet" flag (don't print on failures) to false
        self.quiet = False
        # The messages printed to the *response* buffer by the command
        self.feedbacks: List[Any] = []
        # Start the message queue thread
        self.start()
        # Go through the messages and throw away the initial feedback.
        self.discard_feedback()
        # Stacks for keeping track of the current lemma and module
        self.module_stack: List[str] = []
        self.section_stack: List[str] = []

        # Open the top level module
        if module_name:
            self.run_stmt(f"Module {module_name}.")
        # Execute the commands corresponding to include flags we were
        # passed
        self.exec_includes(includes, prelude)
        # Unset Printing Notations (to get more learnable goals?)
        self.unset_printing_notations()

        # Set up CoqHammer
        self.use_hammer = use_hammer
        if self.use_hammer:
            self.init_hammer()


    @property
    def local_lemmas(self) -> List[str]:
        def generate() -> Iterable[str]:
            for (lemma, is_section) in self._local_lemmas:
                if lemma.startswith(self.module_prefix):
                    yield lemma[len(self.module_prefix):].replace('\n', '')
                else:
                    yield lemma.replace('\n', '')
        return list(generate())

    def cancel_potential_local_lemmas(self, cmd: str) -> None:
        lemmas = self.lemmas_defined_by_stmt(cmd)
        is_section = "Let" in cmd
        for lemma in lemmas:
            self._local_lemmas.remove((lemma, is_section))

    def remove_potential_local_lemmas(self, cmd: str) -> None:
        reset_match = re.match(r"Reset\s+(.*)\.", cmd)
        if reset_match:
            reseted_lemma_name = self.module_prefix + reset_match.group(1)
            for (lemma, is_section) in list(self._local_lemmas):
                if lemma == ":":
                    continue
                lemma_match = re.match(r"\s*([\w'\.]+)\s*:", lemma)
                assert lemma_match, f"{lemma} doesnt match!"
                lemma_name = lemma_match.group(1)
                if lemma_name == reseted_lemma_name:
                    self._local_lemmas.remove((lemma, is_section))
        abort_match = re.match("Abort", cmd)
        if abort_match:
            self._local_lemmas.pop()

    def add_potential_local_lemmas(self, cmd: str) -> None:
        lemmas = self.lemmas_defined_by_stmt(cmd)
        is_section = "Let" in cmd
        for lemma in lemmas:
            self._local_lemmas.append((lemma, is_section))

        for l_idx in range(len(self.local_lemmas)):
            for ol_idx in range(len(self.local_lemmas)):
                if l_idx == ol_idx:
                    continue
                if self.local_lemmas[l_idx][0] == ":":
                    continue
                assert self.local_lemmas[l_idx] != \
                    self.local_lemmas[ol_idx],\
                    self.local_lemmas

    def lemmas_defined_by_stmt(self, cmd: str) -> List[str]:
        cmd = kill_comments(cmd)
        normal_lemma_match = re.match(
            r"\s*(?:" + "|".join(normal_lemma_starting_patterns) +
            r")\s+([\w']*)(.*)",
            cmd,
            flags=re.DOTALL)

        if normal_lemma_match:
            lemma_name = normal_lemma_match.group(1)
            binders, body = unwrap(split_by_char_outside_matching(
                r"\(", r"\)", ":", normal_lemma_match.group(2)))
            if binders.strip():
                lemma_statement = (self.module_prefix + lemma_name +
                                   " : forall " + binders + ", " + body[1:])
            else:
                lemma_statement = self.module_prefix + lemma_name + " " + body
            return [lemma_statement]

        goal_match = re.match(r"\s*(?:Goal)\s+(.*)", cmd, flags=re.DOTALL)

        if goal_match:
            return [": " + goal_match.group(1)]

        morphism_match = re.match(
            r"\s*Add\s+(?:Parametric\s+)?Morphism.*"
            r"with signature(.*)\s+as\s+(\w*)\.",
            cmd, flags=re.DOTALL)
        if morphism_match:
            return [morphism_match.group(2) + " : " + morphism_match.group(1)]

        proposition_match = re.match(r".*Inductive\s*\w+\s*:.*Prop\s*:=(.*)",
                                     cmd, flags=re.DOTALL)
        if proposition_match:
            case_matches = re.finditer(r"\|\s*(\w+\s*:[^|]*)",
                                       proposition_match.group(1))
            constructor_lemmas = [self.module_prefix + case_match.group(1)
                                  for case_match in
                                  case_matches]
            return constructor_lemmas
        obligation_match = re.match(".*Obligation", cmd, flags=re.DOTALL)
        if obligation_match:
            return [":"]

        return []

    @property
    def module_prefix(self) -> str:
        return "".join([module + "." for module in self.module_stack])

    @property
    def cur_lemma(self) -> str:
        return self.local_lemmas[-1]

    @property
    def cur_lemma_name(self) -> str:
        match = re.match(r"\s*([\w'\.]+)\s+:.*", self.cur_lemma)
        assert match, f"Can't match {self.cur_lemma}"
        return match.group(1)

    def tactic_context(self, relevant_lemmas) -> TacticContext:
        return TacticContext(relevant_lemmas,
                             self.prev_tactics,
                             self.hypotheses,
                             self.goals)

    # Hammer prints a lot of stuff when it gets imported. Discard all of it.
    def init_hammer(self):
        self.hammer_timeout = 100
        # atp_limit = 29 * self.hammer_timeout // 60
        # reconstr_limit = 28 * self.hammer_timeout // 60
        # crush_limit = 3 * self.hammer_timeout // 60
        # hammer_cmd = "(Add () \"From Hammer Require Import Hammer. Set Hammer ATPLimit %d. Set Hammer ReconstrLimit %d. Set Hammer CrushLimit %d.\")" % (atp_limit, reconstr_limit, crush_limit)
        hammer_cmd = "(Add () \"From Hammer Require Import Hammer.\")"
        self.send_acked(hammer_cmd)
        self.discard_feedback()
        self.discard_feedback()
        self.update_state()
        self.get_completed()

    # Send some text to serapi, and flush the stream to make sure they
    # get it. NOT FOR EXTERNAL USE
    def send_flush(self, cmd: str):
        assert self._fin
        eprint("SENT: " + cmd, guard=self.verbose >= 4)
        if self.log_outgoing_messages:
            with self.log_outgoing_messages.open('w') as f:
                print(cmd, file=f)
        try:
            self._fin.write(cmd.encode('utf-8'))
            self._fin.flush()
        except BrokenPipeError:
            raise CoqAnomaly("Coq process unexpectedly quit. Possibly running "
                             "out of memory due to too many threads?")

    def send_acked(self, cmd: str):
        self.send_flush(cmd)
        self.get_ack()

    def ask(self, cmd: str, complete: bool = True):
        return loads(self.ask_text(cmd, complete))

    def ask_text(self, cmd: str, complete: bool = True):
        assert self.message_queue.empty(), self.messages
        self.send_acked(cmd)
        msg = self.get_message_text(complete)
        return msg

    @property
    def messages(self):
        return [dumps(msg) for msg in list(self.message_queue.queue)]

    def get_hammer_premise_names(self, k: int) -> List[str]:
        if not self.goals:
            return []
        try:
            oldquiet = self.quiet
            self.quiet = True
            self.run_stmt(f"predict {k}.", timeout=120)
            self.quiet = oldquiet
            premise_names = self.feedbacks[3][1][3][1][3][1].split(", ")
            self.cancel_last()
            return premise_names
        except CoqExn:
            return []

    def get_hammer_premises(self, k: int = 10) -> List[str]:
        old_timeout = self.timeout
        self.timeout = 600
        names = self.get_hammer_premise_names(k)

        def get_full_line(name: str) -> str:
            try:
                self.send_acked(f"(Query () (Vernac \"Check {name}.\"))")
                try:
                    nextmsg = self.get_message()
                except TimeoutError:
                    eprint("Timed out waiting for initial message")
                while match(normalizeMessage(nextmsg),
                            ["Feedback", [["doc_id", int], ["span_id", int],
                                          ["route", int],
                                          ["contents", "Processed"]]],
                            lambda *args: True,
                            _,
                            lambda *args: False):
                    try:
                        nextmsg = self.get_message()
                    except TimeoutError:
                        eprint("Timed out waiting for message")
                pp_term = nextmsg[1][3][1][3]
                try:
                    nextmsg = self.get_message()
                except TimeoutError:
                    eprint("Timed out waiting for message")
                match(normalizeMessage(nextmsg),
                      ["Answer", int, ["ObjList", []]],
                      lambda *args: None,
                      _, lambda *args: raise_(UnrecognizedError(nextmsg)))
                try:
                    self.get_completed()
                except TimeoutError:
                    eprint("Timed out waiting for completed message")
                try:
                    result = re.sub(r"\s+", " ", self.ppToTermStr(pp_term))
                except TimeoutError:
                    eprint("Timed out when converting ppterm")
                return result
            except TimeoutError:
                eprint("Timed out when getting full line!")
                return ""
        full_lines = [line for line in
                      [get_full_line(name) for name in names]
                      if line]
        self.timeout = old_timeout
        return full_lines

    # Run a command. This is the main api function for this
    # class. Sends a single command to the running serapi
    # instance. Returns nothing: if you want a response, call one of
    # the other methods to get it.
    def run_stmt(self, stmt: str, timeout: Optional[int] = None):
        if timeout:
            old_timeout = self.timeout
            self.timeout = timeout
        self.flush_queue()
        eprint("Running statement: " + stmt.lstrip('\n'),
               guard=self.verbose >= 2)  # lstrip makes output shorter
        # We need to escape some stuff so that it doesn't get stripped
        # too early.
        stmt = stmt.replace("\\", "\\\\")
        stmt = stmt.replace("\"", "\\\"")
        # Kill the comments early so we can recognize comments earlier
        stmt = kill_comments(stmt)
        # We'll wrap the actual running in a try block so that we can
        # report which command the error came from at this
        # level. Other higher level code might re-catch it.
        history_len_before = len(self.tactic_history.getFullHistory())
        context_before = self.proof_context
        try:
            # Preprocess_command sometimes turns one command into two,
            # to get around some limitations of the serapi interface.
            for stm in preprocess_command(stmt):
                self.add_potential_module_stack_cmd(stm)
                # Get initial context
                # Send the command
                assert self.message_queue.empty(), self.messages
                self.send_acked("(Add () \"{}\")\n".format(stm))
                # Get the response, which indicates what state we put
                # serapi in.
                self.update_state()
                self.get_completed()
                assert self.message_queue.empty()

                # Execute the statement.
                self.send_acked("(Exec {})\n".format(self.cur_state))
                # Finally, get the result of the command
                self.feedbacks = self.get_feedbacks()
                # Get a new proof context, if it exists
                if stm.strip == "{":
                    self.get_enter_goal_context()
                else:
                    self.get_proof_context()

                if not context_before and self.proof_context:
                    self.add_potential_local_lemmas(stm)
                elif not self.proof_context:
                    self.remove_potential_local_lemmas(stm)
                    self.tactic_history = TacticHistory()

                # Manage the tactic history
                if possibly_starting_proof(stm) and self.proof_context:
                    self.tactic_history.addTactic(stm)
                elif re.match(r"\s*(?:\d+\s*:)?\s*[{]\s*", stm):
                    assert context_before
                    self.tactic_history.openSubgoal(
                        context_before.fg_goals[1:])
                elif re.match(r"\s*[}]\s*", stm):
                    self.tactic_history.closeSubgoal()
                elif self.proof_context:
                    # If we saw a new proof context, we're still in a
                    # proof so append the command to our prev_tactics
                    # list.
                    self.tactic_history.addTactic(stm)

        # If we hit a problem let the user know what file it was in,
        # and then throw it again for other handlers. NOTE: We may
        # want to make this printing togglable (at this level), since
        # sometimes errors are expected.
        except (CoqExn, BadResponse, AckError,
                CompletedError, TimeoutError) as e:
            self.handle_exception(e, stmt)
        finally:
            if self.proof_context and self.verbose >= 3:
                eprint(
                    f"History is now {self.tactic_history.getFullHistory()}")
                summarizeContext(self.proof_context)
            assert len(self.tactic_history.getFullHistory()) == \
                history_len_before + 1 or \
                (re.match(r"(?:\d+\s*:)?\s*{", stmt.strip()) and
                 len(self.tactic_history.getFullHistory()) ==
                 history_len_before + 2) or \
                (stmt.strip() == "}" and
                 len(self.tactic_history.getFullHistory()) ==
                 history_len_before) or \
                self.proof_context == context_before or \
                stmt.strip() == "Proof." or \
                (self.proof_context is None and ending_proof(stmt))
            if timeout:
                self.timeout = old_timeout

    @property
    def prev_tactics(self):

        return self.tactic_history.getCurrentHistory()

    def handle_exception(self, e: SerapiException, stmt: str):
        eprint("Problem running statement: {}\n".format(stmt),
               guard=(not self.quiet or self.verbose >= 2))
        match(e,
              TimeoutError,
              lambda *args: progn(self.cancel_failed(),  # type: ignore
                                  raise_(TimeoutError(
                                      "Statment \"{}\" timed out."
                                      .format(stmt)))),
              _, lambda e: None)
        coqexn_msg = match(normalizeMessage(e.msg),
                           ['Answer', int, ['CoqExn', TAIL]],
                           lambda sentence_num, rest:
                           "\n".join(searchStrsInMsg(rest)),
                           str, lambda s: s,
                           [str], lambda s: s,
                           _, None)
        if coqexn_msg:
            eprint(coqexn_msg, guard=(not self.quiet or self.verbose >= 2))
            if ("Stream\\.Error" in coqexn_msg
                    or "Syntax error" in coqexn_msg
                    or "Syntax Error" in coqexn_msg):
                self.get_completed()
                raise ParseError(f"Couldn't parse command {stmt}")
            elif "CLexer.Error" in coqexn_msg:
                self.get_completed()
                raise ParseError(f"Couldn't parse command {stmt}")
            elif "NoSuchGoals" in coqexn_msg:
                self.get_completed()
                self.cancel_failed()
                raise NoSuchGoalError("")
            elif "Invalid_argument" in coqexn_msg:
                raise ParseError(f"Invalid argument in {stmt}")
            elif "Not_found" in coqexn_msg:
                self.get_completed()
                self.cancel_failed()
                raise e
            elif "Overflowed" in coqexn_msg or "Stack overflow" in coqexn_msg:
                self.get_completed()
                raise CoqAnomaly("Overflowed")
            elif "Anomaly" in coqexn_msg:
                self.get_completed()
                raise CoqAnomaly(coqexn_msg)
            else:
                self.get_completed()
                self.cancel_failed()
                raise CoqExn(coqexn_msg)
        else:
            match(normalizeMessage(e.msg),
                  ['Stream\\.Error', str],
                  lambda *args: progn(self.get_completed(),
                                      raise_(ParseError(
                                          "Couldn't parse command {}"
                                          .format(stmt)))),

                  ['CErrors\\.UserError', _],
                  lambda inner: progn(self.get_completed(),
                                      self.cancel_failed(),  # type: ignore
                                      raise_(e)),
                  ['ExplainErr\\.EvaluatedError', TAIL],
                  lambda inner: progn(self.get_completed(),
                                      self.cancel_failed(),  # type: ignore
                                      raise_(e)),
                  _, lambda *args: progn(raise_(UnrecognizedError(args))))

    # Flush all messages in the message queue
    def flush_queue(self) -> None:
        while not self.message_queue.empty():
            self.get_message()

    def ppStrToTermStr(self, pp_str: str) -> str:
        answer = self.ask(
            f"(Print ((pp ((pp_format PpStr)))) (CoqPp {pp_str}))")
        return match(normalizeMessage(answer),
                     ["Answer", int, ["ObjList", [["CoqString", _]]]],
                     lambda statenum, s: str(s),
                     ["Answer", int, ["CoqExn", TAIL]],
                     lambda statenum, msg:
                     raise_(CoqExn(searchStrsInMsg(msg))))

    def ppToTermStr(self, pp) -> str:
        return self.ppStrToTermStr(dumps(pp))

    @functools.lru_cache(maxsize=128)
    def sexpStrToTermStr(self, sexp_str: str) -> str:
        answer = self.ask(
            f"(Print ((pp ((pp_format PpStr)))) (CoqConstr {sexp_str}))")
        return match(normalizeMessage(answer),
                     ["Answer", int, ["ObjList", [["CoqString", _]]]],
                     lambda statenum, s: str(s),
                     ["Answer", int, ["CoqExn", TAIL]],
                     lambda statenum, msg:
                     raise_(CoqExn(searchStrsInMsg(msg))))

    def sexpToTermStr(self, sexp) -> str:
        return self.sexpStrToTermStr(dumps(sexp))

    def parseSexpHypStr(self, sexp_str: str) -> str:
        var_sexps_str, mid_str, term_sexp_str = \
            cast(List[str], parseSexpOneLevel(sexp_str))

        def get_id(var_pair_str: str) -> str:
            id_possibly_quoted = unwrap(re.match(
                r"\(Id\s*(.*)\)", var_pair_str)).group(1)
            if id_possibly_quoted[0] == "\"" and \
               id_possibly_quoted[-1] == "\"":
                return id_possibly_quoted[1:-1]
            return id_possibly_quoted
        ids_str = ",".join([get_id(var_pair_str) for
                            var_pair_str in
                            cast(List[str], parseSexpOneLevel(var_sexps_str))])
        term_str = self.sexpStrToTermStr(term_sexp_str)
        return f"{ids_str} : {term_str}"

    def parseSexpHyp(self, sexp) -> str:
        var_sexps, _, term_sexp = sexp
        ids_str = ",".join([dumps(var_sexp[1]) for var_sexp in var_sexps])
        term_str = self.sexpToTermStr(term_sexp)
        return f"{ids_str} : {term_str}"

    def parseSexpGoalStr(self, sexp_str: str) -> Obligation:
        goal_match = goal_regex.fullmatch(sexp_str)
        assert goal_match, sexp_str + "didn't match"
        goal_num_str, goal_term_str, hyps_list_str = \
            goal_match.group(1, 2, 3)
        goal_str = self.sexpStrToTermStr(goal_term_str).replace(r"\.", ".")
        hyps = [self.parseSexpHypStr(hyp_str) for hyp_str in
                cast(List[str], parseSexpOneLevel(hyps_list_str))]
        return Obligation(hyps, goal_str)

    def parseSexpGoal(self, sexp) -> Obligation:
        goal_num, goal_term, hyps_list = \
            match(normalizeMessage(sexp),
                  [["name", int], ["ty", _], ["hyp", list]],
                  lambda *args: args)
        goal_str = self.sexpToTermStr(goal_term)
        hyps = [self.parseSexpHyp(hyp_sexp) for hyp_sexp in hyps_list]
        return Obligation(hyps, goal_str)

    def parseBgGoal(self, sexp) -> Obligation:
        return match(normalizeMessage(sexp),
                     [[], [_]],
                     lambda inner_sexp: self.parseSexpGoal(inner_sexp))

    # Cancel the last command which was sucessfully parsed by
    # serapi. Even if the command failed after parsing, this will
    # still cancel it. You need to call this after a command that
    # fails after parsing, but not if it fails before.
    def cancel_last(self) -> None:
        context_before = self.proof_context
        if self.proof_context:
            if len(self.tactic_history.getFullHistory()) > 0:
                cancelled = self.tactic_history.getNextCancelled()
                eprint(f"Cancelling {cancelled} "
                       f"from state {self.cur_state}",
                       guard=self.verbose)
                self.cancel_potential_local_lemmas(cancelled)
            else:
                eprint("Cancelling something (not in history)",
                       guard=self.verbose)
        else:
            cancelled = ""
            eprint(f"Cancelling vernac "
                   f"from state {self.cur_state}",
                   guard=self.verbose)
        self.__cancel()

        # Fix up the previous tactics
        if context_before and len(self.tactic_history.getFullHistory()) > 0:
            self.tactic_history.removeLast(context_before.fg_goals)
        if not self.proof_context:
            assert len(self.tactic_history.getFullHistory()) == 0, \
                ("History is desynced!", self.tactic_history.getFullHistory())
            self.tactic_history = TacticHistory()
        assert self.message_queue.empty(), self.messages
        if self.proof_context and self.verbose >= 3:
            eprint(f"History is now {self.tactic_history.getFullHistory()}")
            summarizeContext(self.proof_context)

    def __cancel(self) -> None:
        self.flush_queue()
        assert self.message_queue.empty(), self.messages
        # Run the cancel
        self.send_acked("(Cancel ({}))".format(self.cur_state))
        # Get the response from cancelling
        self.cur_state = self.get_cancelled()
        # Get a new proof context, if it exists
        self.get_proof_context()

    def cancel_failed(self) -> None:
        self.__cancel()

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

    def add_lib(self, origpath: str, logicalpath: str) -> None:
        addStm = ("(Add () \"Add LoadPath \\\"{}\\\" as {}.\")\n"
                  .format(origpath, logicalpath))
        self.send_acked(addStm)
        self.update_state()
        self.get_completed()
        self.send_acked("(Exec {})\n".format(self.cur_state))
        self.discard_feedback()
        self.discard_feedback()
        self.get_completed()

    def add_ocaml_lib(self, path: str) -> None:
        addStm = ("(Add () \"Add ML Path \\\"{}\\\".\")\n"
                  .format(path))
        self.send_acked(addStm)
        self.update_state()
        self.get_completed()
        self.send_acked("(Exec {})\n".format(self.cur_state))
        self.discard_feedback()
        self.discard_feedback()
        self.get_completed()

    def add_lib_rec(self, origpath: str, logicalpath: str) -> None:
        addStm = ("(Add () \"Add Rec LoadPath \\\"{}\\\" as {}.\")\n"
                  .format(origpath, logicalpath))
        self.send_acked(addStm)
        self.update_state()
        self.get_completed()
        self.send_acked("(Exec {})\n".format(self.cur_state))
        self.discard_feedback()
        self.discard_feedback()
        self.get_completed()

    def search_about(self, symbol: str) -> List[str]:
        self.send_acked(f"(Query () (Vernac \"Search {symbol}.\"))")
        lemma_msgs: List[str] = []
        nextmsg = self.get_message()
        while match(normalizeMessage(nextmsg),
                    ["Feedback", [["doc_id", int], ["span_id", int],
                                  ["route", int],
                                  ["contents", ["ProcessingIn", str]]]],
                    lambda *args: True,
                    ["Feedback", [["doc_id", int], ["span_id", int],
                                  ["route", int],
                                  ["contents", "Processed"]]],
                    lambda *args: True,
                    _,
                    lambda *args: False):
            nextmsg = self.get_message()
        while match(normalizeMessage(nextmsg),
                    ["Feedback", [["doc_id", int], ["span_id", int],
                                  ["route", int],
                                  ["contents", ["Message", "Notice",
                                                [], TAIL]]]],
                    lambda *args: True,
                    _, lambda *args: False):
            oldmsg = nextmsg
            try:
                nextmsg = self.get_message()
                lemma_msgs.append(oldmsg)
            except RecursionError:
                pass
        self.get_completed()
        str_lemmas = [re.sub(r"\s+", " ",
                             self.ppToTermStr(lemma_msg[1][3][1][3]))
                      for lemma_msg in lemma_msgs[:10]]
        return str_lemmas

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

    def exec_includes(self, includes_string: str, prelude: str) -> None:
        for rmatch in re.finditer(r"-R\s*(\S*)\s*(\S*)\s*", includes_string):
            self.add_lib_rec("./" + rmatch.group(1), rmatch.group(2))
        for qmatch in re.finditer(r"-Q\s*(\S*)\s*(\S*)\s*", includes_string):
            self.add_lib("./" + qmatch.group(1), qmatch.group(2))
        for imatch in re.finditer(r"-I\s*(\S*)", includes_string):
            self.add_ocaml_lib("./" + imatch.group(1))

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
                           ["CoqExn", TAIL],
                           lambda rest:
                           raise_(CoqExn("\n".join(searchStrsInMsg(rest)))),
                           ["Added", int, TAIL],
                           lambda state_num, tail: state_num),
                     _, lambda x: raise_(BadResponse(msg)))

    def discard_feedback(self) -> None:
        try:
            feedback_message = self.get_message()
            while feedback_message[1][3][1] != Symbol("Processed"):
                feedback_message = self.get_message()
        except TimeoutError:
            pass
        except CoqAnomaly as e:
            if e.msg != "Timing Out":
                raise

    def discard_initial_feedback(self) -> None:
        feedback1 = self.get_message()
        feedback2 = self.get_message()
        match(normalizeMessage(feedback1), ["Feedback", TAIL],
              lambda *args: None,
              _, lambda *args: raise_(BadResponse(feedback1)))
        match(normalizeMessage(feedback2), ["Feedback", TAIL],
              lambda *args: None,
              _, lambda *args: raise_(BadResponse(feedback2)))

    def interrupt(self) -> None:
        self._proc.send_signal(signal.SIGINT)
        self.flush_queue()

    def get_message(self, complete=False) -> Any:
        msg_text = self.get_message_text(complete=complete)
        assert msg_text != "None", msg_text
        try:
            return loads(msg_text, nil=None)
        except ExpectClosingBracket:
            eprint(
                f"Tried to load a message but it's ill formed! \"{msg_text}\"")
            raise CoqAnomaly("")

    def get_message_text(self, complete=False) -> Any:
        try:
            msg = self.message_queue.get(timeout=self.timeout)
            if complete:
                self.get_completed()
            assert msg is not None
            return msg
        except queue.Empty:
            eprint("Command timed out! Interrupting", guard=self.verbose)
            self._proc.send_signal(signal.SIGINT)
            num_breaks = 1
            try:
                interrupt_response = \
                    loads(self.message_queue.get(timeout=self.timeout))
            except queue.Empty:
                self._proc.send_signal(signal.SIGINT)
                num_breaks += 1
                try:
                    interrupt_response = \
                        loads(self.message_queue.get(timeout=self.timeout))
                except queue.Empty:
                    raise CoqAnomaly("Timing Out")

            got_answer_after_interrupt = match(
                normalizeMessage(interrupt_response),
                ["Answer", int, ["CoqExn", TAIL]],
                lambda *args: False,
                ["Answer", TAIL],
                lambda *args: True,
                _, lambda *args: False)
            if got_answer_after_interrupt:
                self.get_completed()
                for i in range(num_breaks):
                    try:
                        after_interrupt_msg = loads(self.message_queue.get(
                            timeout=self.timeout))
                    except queue.Empty:
                        raise CoqAnomaly("Timing out")
                    assert isBreakMessage(after_interrupt_msg), \
                        after_interrupt_msg
                assert self.message_queue.empty(), self.messages
                return dumps(interrupt_response)
            else:
                for i in range(num_breaks):
                    try:
                        after_interrupt_msg = loads(self.message_queue.get(
                            timeout=self.timeout))
                    except queue.Empty:
                        raise CoqAnomaly("Timing out")
                self.get_completed()
                assert self.message_queue.empty(), self.messages
                raise TimeoutError("")
            assert False, (interrupt_response, self.messages)

    def get_feedbacks(self) -> List['Sexp']:
        feedbacks = []  # type: List[Sexp]
        next_message = self.get_message()
        while(isinstance(next_message, list) and
              next_message[0] == Symbol("Feedback")):
            feedbacks.append(next_message)
            next_message = self.get_message()
        fin = next_message
        match(normalizeMessage(fin),
              ["Answer", _, "Completed", TAIL], lambda *args: None,
              ['Answer', _, ["CoqExn", [_, _, _, _, _, ['str', _]]]],
              lambda statenum, loc1, loc2, loc3, loc4, loc5, inner:
              raise_(CoqExn(fin)),
              _, lambda *args: progn(eprint(f"message is \"{repr(fin)}\""),
                                     raise_(UnrecognizedError(fin))))

        return feedbacks

    def count_fg_goals(self) -> int:
        if not self.proof_context:
            return 0
        return len(self.proof_context.fg_goals)

    def get_cancelled(self) -> int:
        try:
            feedback = self.get_message()

            new_statenum = \
                match(normalizeMessage(feedback),
                      ["Answer", int, ["CoqExn", TAIL]],
                      lambda docnum, rest:
                      raise_(CoqAnomaly("Overflowed"))
                      if "Stack overflow" in "\n".join(searchStrsInMsg(rest))
                      else raise_(CoqExn(feedback)),
                      ["Feedback", [['doc_id', int], ['span_id', int], TAIL]],
                      lambda docnum, statenum, *rest: statenum,
                      _, lambda *args: raise_(BadResponse(feedback)))

            cancelled_answer = self.get_message()

            match(normalizeMessage(cancelled_answer),
                  ["Answer", int, ["Canceled", list]],
                  lambda _, statenums: min(statenums),
                  ["Answer", int, ["CoqExn", TAIL]],
                  lambda statenum, rest:
                  raise_(CoqExn("\n".join(searchStrsInMsg(rest)))),
                  _, lambda *args: raise_(BadResponse(cancelled_answer)))
        finally:
            self.get_completed()

        return new_statenum

    def extract_proof_context(self, raw_proof_context: 'Sexp') -> str:
        assert isinstance(raw_proof_context, list), raw_proof_context
        assert len(raw_proof_context) > 0, raw_proof_context
        assert isinstance(raw_proof_context[0], list), raw_proof_context
        return cast(List[List[str]], raw_proof_context)[0][1]

    @property
    def goals(self) -> str:
        if self.proof_context and self.proof_context.fg_goals:
            return self.proof_context.fg_goals[0].goal
        else:
            return ""

    @property
    def hypotheses(self) -> List[str]:
        if self.proof_context and self.proof_context.fg_goals:
            return self.proof_context.fg_goals[0].hypotheses
        else:
            return []

    def get_enter_goal_context(self) -> None:
        assert self.proof_context
        self.proof_context = ProofContext([self.proof_context.fg_goals[0]],
                                          self.proof_context.bg_goals +
                                          self.proof_context.fg_goals[1:],
                                          self.proof_context.shelved_goals,
                                          self.proof_context.given_up_goals)

    def get_proof_context(self) -> None:
        # Try to do this the right way, fall back to the
        # wrong way if we run into this bug:
        # https://github.com/ejgallego/coq-serapi/issues/150
        try:
            text_response = self.ask_text("(Query () Goals)")
            context_match = re.fullmatch(
                r"\(Answer\s+\d+\s*\(ObjList\s*(.*)\)\)\n",
                text_response)
            if not context_match:
                if "Stack overflow" in text_response:
                    raise CoqAnomaly(f"\"{text_response}\"")
                else:
                    raise BadResponse(f"\"{text_response}\"")
            context_str = context_match.group(1)
            if context_str == "()":
                self.proof_context = None
            else:
                goals_match = re.match(r"\(\(CoqGoal\s*"
                                       r"\(\(goals\s*(.*)\)"
                                       r"\(stack\s*(.*)\)"
                                       r"\(shelf\s*(.*)\)"
                                       r"\(given_up\s*(.*)\)"
                                       r"\(bullet\s*.*\)\)\)\)",
                                       context_str)
                if not goals_match:
                    raise BadResponse(context_str)
                fg_goals_str, bg_goals_str, \
                    shelved_goals_str, given_up_goals_str = \
                    goals_match.groups()
                unparsed_levels = cast(List[str],
                                       parseSexpOneLevel(bg_goals_str))
                parsed2 = [uuulevel
                           for ulevel in unparsed_levels
                           for uulevel in cast(List[str],
                                               parseSexpOneLevel(ulevel))
                           for uuulevel in cast(List[str],
                                                parseSexpOneLevel(uulevel))]
                bg_goals = [self.parseSexpGoalStr(bg_goal_str)
                            for bg_goal_str in parsed2]
                self.proof_context = ProofContext(
                    [self.parseSexpGoalStr(goal)
                     for goal in cast(List[str],
                                      parseSexpOneLevel(fg_goals_str))],
                    bg_goals,
                    [self.parseSexpGoalStr(shelved_goal)
                     for shelved_goal in
                     cast(List[str], parseSexpOneLevel(shelved_goals_str))],
                    [self.parseSexpGoalStr(given_up_goal)
                     for given_up_goal in
                     cast(List[str], parseSexpOneLevel(given_up_goals_str))])
        except CoqExn:
            self.send_acked("(Query ((pp ((pp_format PpStr)))) Goals)")

            msg = self.get_message()
            proof_context_msg = match(
                normalizeMessage(msg),
                ["Answer", int, ["CoqExn", TAIL]],
                lambda statenum, rest:
                raise_(CoqAnomaly("Stack overflow")) if
                "Stack overflow." in searchStrsInMsg(rest) else
                raise_(CoqExn(searchStrsInMsg(rest))),
                ["Answer", int, list],
                lambda statenum, contents: contents,
                _, lambda *args:
                raise_(UnrecognizedError(dumps(msg))))
            self.get_completed()
            if len(proof_context_msg) == 0:
                self.proof_context = None
            else:
                newcontext = self.extract_proof_context(proof_context_msg[1])
                if newcontext == "none":
                    self.proof_context = ProofContext([], [], [], [])
                else:
                    self.proof_context = \
                        ProofContext(
                            [parsePPSubgoal(substr) for substr
                             in re.split(r"\n\n|(?=\snone)", newcontext)
                             if substr.strip()],
                            [], [], [])

    def get_lemmas_about_head(self) -> List[str]:
        if self.goals.strip() == "":
            return []
        goal_head = self.goals.split()[0]
        if (goal_head == "forall"):
            return []
        answer = self.search_about(goal_head)
        assert self.message_queue.empty(), self.messages
        return answer

    def run_into_next_proof(self, commands: List[str]) \
            -> Optional[Tuple[List[str], List[str]]]:
        assert not self.proof_context, "We're already in a proof"
        commands_iter = iter(commands)
        commands_run = []
        for command in commands_iter:
            self.run_stmt(command, timeout=60)
            commands_run.append(command)
            if self.proof_context:
                return list(commands_iter), commands_run
        return [], commands_run

    def finish_proof(self, commands: List[str]) \
            -> Optional[Tuple[List[str], List[str]]]:
        assert self.proof_context, "We're already out of a proof"
        commands_iter = iter(commands)
        commands_run = []
        for command in commands_iter:
            self.run_stmt(command, timeout=60)
            commands_run.append(command)
            if not self.proof_context:
                return list(commands_iter), commands_run
        return None

    def run(self) -> None:
        assert self._fout
        while(True):
            try:
                line = self._fout.readline().decode('utf-8')
            except ValueError:
                continue
            if line == '':
                break
            self.message_queue.put(line)
            eprint(f"RECEIVED: {line}", guard=self.verbose >= 4)

    def add_potential_module_stack_cmd(self, cmd: str) -> None:
        stripped_cmd = kill_comments(cmd).strip()
        module_start_match = re.match(
            r"Module\s+(?:Import\s+)?(?:Type\s+)?([\w']*)", stripped_cmd)
        if stripped_cmd.count(":=") > stripped_cmd.count("with"):
            module_start_match = None
        section_start_match = re.match(r"Section\s+([\w']*)\b(?!.*:=)",
                                       stripped_cmd)
        end_match = re.match(r"End (\w*)\.", stripped_cmd)
        if module_start_match:
            self.module_stack.append(module_start_match.group(1))
        elif section_start_match:
            self.section_stack.append(section_start_match.group(1))
        elif end_match:
            if self.module_stack and \
               self.module_stack[-1] == end_match.group(1):
                self.module_stack.pop()
            elif self.section_stack and \
                    self.section_stack[-1] == end_match.group(1):
                self._local_lemmas = \
                    [(lemma, is_section) for (lemma, is_section)
                     in self._local_lemmas if not is_section]
                self.section_stack.pop()
            else:
                assert False, \
                    f"Unrecognized End \"{cmd}\", " \
                    f"top of module stack is {self.module_stack[-1]}"

    def kill(self) -> None:
        assert self._proc.stdout
        self._proc.terminate()
        self._proc.stdout.close()
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.kill()
        threading.Thread.join(self)
    pass


goal_regex = re.compile(r"\(\(info\s*\(\(evar\s*\(Ser_Evar\s*(\d+)\)\)"
                        r"\(name\s*\((?:\(Id\s*[\w']+\))*\)\)\)\)"
                        r"\(ty\s*(.*)\)\s*\(hyp\s*(.*)\)\)")


def isBreakMessage(msg: 'Sexp') -> bool:
    return match(normalizeMessage(msg),
                 "Sys\\.Break", lambda *args: True,
                 _, lambda *args: False)


def isBreakAnswer(msg: 'Sexp') -> bool:
    return "Sys\\.Break" in searchStrsInMsg(normalizeMessage(msg))


@contextlib.contextmanager
def SerapiContext(coq_commands: List[str], module_name: str,
                  prelude: str, use_hammer: bool = False,
                  log_outgoing_messages: Optional[Path2] = None) \
                  -> Iterator[Any]:
    coq = SerapiInstance(coq_commands, module_name, prelude,
                         use_hammer=use_hammer,
                         log_outgoing_messages=log_outgoing_messages)
    try:
        yield coq
    finally:
        coq.kill()


normal_lemma_starting_patterns = [
    r"(?:Local|Global\s+)?Lemma",
    "Coercion",
    "Theorem",
    "Remark",
    "Proposition",
    "Definition",
    "Program Definition",
    "Example",
    "Fixpoint",
    "Corollary",
    "Let",
    r"(?<!Declare\s)Instance",
    "Global Instance",
    "Local Instance",
    "Function",
    "Property"]
special_lemma_starting_patterns = [
    "Derive",
    "Goal",
    "Add Morphism",
    "Next Obligation",
    r"Obligation\s+\d+",
    "Add Parametric Morphism"]

lemma_starting_patterns = \
    normal_lemma_starting_patterns + special_lemma_starting_patterns


def possibly_starting_proof(command: str) -> bool:
    stripped_command = kill_comments(command).strip()
    return bool(re.match("(" + "|".join(lemma_starting_patterns) + r")\s*",
                         stripped_command))


def ending_proof(command: str) -> bool:
    stripped_command = kill_comments(command).strip()
    return ("Qed" in stripped_command or
            "Defined" in stripped_command or
            "Admitted" in stripped_command or
            "Abort" in stripped_command or
            (re.match(r"\s*Proof\s+\S+\s*", stripped_command) is not None and
             re.match(r"\s*Proof\s+with", stripped_command) is None))


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


def next_proof(cmds: Iterator[str]) -> Iterator[str]:
    next_cmd = next(cmds)
    assert possibly_starting_proof(next_cmd), next_cmd
    while not ending_proof(next_cmd):
        yield next_cmd
        try:
            next_cmd = next(cmds)
        except StopIteration:
            return
    yield next_cmd


def preprocess_command(cmd: str) -> List[str]:
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
        match = re.fullmatch(r"\s*Require(\s+(?:(?:Import)|(?:Export)))?"
                             r"((?:\s+\S+)*)\s+({})\s*((?:\s+\S*)*)\.\s*"
                             .format(lib), cmd)
        if match:
            if match.group(1):
                impG = match.group(1)
            else:
                impG = ""
            if match.group(4):
                after = match.group(4)
            else:
                after = ""
            if (re.fullmatch(r"\s*", match.group(2)) and
                    re.fullmatch(r"\s*", after)):
                return ["From Coq Require" + impG + " " + match.group(3) + "."]
            else:
                return ["From Coq Require" + impG + " " + match.group(3) + "."
                        ] + preprocess_command("Require " + impG.strip() + " "
                                               + match.group(2).strip() + " "
                                               + after + ".")
    return [cmd] if cmd.strip() else []


def get_stem(tactic: str) -> str:
    return split_tactic(tactic)[0]


def split_tactic(tactic: str) -> Tuple[str, str]:
    tactic = kill_comments(tactic).strip()
    if not tactic:
        return ("", "")
    if re.match(r"^\s*[-+*\{\}]+\s*$", tactic):
        stripped = tactic.strip()
        return stripped[:-1], stripped[-1]
    if split_by_char_outside_matching(r"\(", r"\)", ";", tactic):
        return tactic, ""
    for prefix in ["try", "now", "repeat", "decide"]:
        prefix_match = re.match(r"{}\s+(.*)".format(prefix), tactic)
        if prefix_match:
            rest_stem, rest_rest = split_tactic(prefix_match.group(1))
            return prefix + " " + rest_stem, rest_rest
    for special_stem in ["rewrite <-", "rewrite !",
                         "intros until", "simpl in"]:
        special_match = re.match(r"{}\s*(.*)".format(special_stem), tactic)
        if special_match:
            return special_stem, special_match.group(1)
    match = re.match(r"^\(?(\w+)(\W+.*)?", tactic)
    assert match, "tactic \"{}\" doesn't match!".format(tactic)
    stem, rest = match.group(1, 2)
    if not rest:
        rest = ""
    return stem, rest


def parse_hyps(hyps_str: str) -> List[str]:
    lets_killed = kill_nested(r"\Wlet\s", r"\sin\s", hyps_str)
    funs_killed = kill_nested(r"\Wfun\s", "=>", lets_killed)
    foralls_killed = kill_nested(r"\Wforall\s", ",", funs_killed)
    fixs_killed = kill_nested(r"\Wfix\s", ":=", foralls_killed)
    structs_killed = kill_nested(r"\W\{\|\s", r"\|\}", fixs_killed)
    hyps_replaced = re.sub(":=.*?:(?!=)", ":", structs_killed, flags=re.DOTALL)
    var_terms = re.findall(r"(\S+(?:, \S+)*) (?::=.*?)?:(?!=)\s.*?",
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
        assert re.search(":(?!=)", hyp) is not None, \
            "hyp: {}, hyps_str: {}\nhyps_list: {}\nvar_terms: {}"\
            .format(hyp, hyps_str, hyps_list, var_terms)
    return hyps_list


def kill_nested(start_string: str, end_string: str, hyps: str) \
        -> str:
    def searchpos(pattern: str, hyps: str, end: bool = False):
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
    while (next_forall_pos != float("Inf") or
           (next_comma_pos != float("Inf") and forall_depth > 0)):
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
            searchpos(end_string, hyps[cur_position+1:], end=True) + \
            cur_position + 1
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


def get_var_term_in_hyp(hyp: str) -> str:
    return hyp.partition(":")[0].strip()


def get_hyp_type(hyp: str) -> str:
    if re.search(":(?!=)", hyp) is None:
        return ""
    return re.split(":(?!=)", hyp, maxsplit=1)[1].strip()


def get_vars_in_hyps(hyps: List[str]) -> List[str]:
    var_terms = [get_var_term_in_hyp(hyp) for hyp in hyps]
    var_names = [name.strip() for term in var_terms
                 for name in term.split(",")]
    return var_names


def get_indexed_vars_in_hyps(hyps: List[str]) -> List[Tuple[str, int]]:
    var_terms = [get_var_term_in_hyp(hyp) for hyp in hyps]
    var_names = [(name.strip(), hyp_idx)
                 for hyp_idx, term in enumerate(var_terms)
                 for name in term.split(",")]
    return var_names


def get_indexed_vars_dict(hyps: List[str]) -> Dict[str, int]:
    result = {}
    for hyp_var, hyp_idx in get_indexed_vars_in_hyps(hyps):
        if hyp_var not in result:
            result[hyp_var] = hyp_idx
    return result


def get_first_var_in_hyp(hyp: str) -> str:
    return get_var_term_in_hyp(hyp).split(",")[0].strip()


def normalizeMessage(sexp, depth: int = 5):
    if depth <= 0:
        return sexp
    if isinstance(sexp, list):
        return [normalizeMessage(item, depth=depth-1) for item in sexp]
    if isinstance(sexp, Symbol):
        return dumps(sexp)
    else:
        return sexp


def tacticTakesHypArgs(stem: str) -> bool:
    now_match = re.match(r"\s*now\s+(.*)", stem)
    if now_match:
        return tacticTakesHypArgs(now_match.group(1))
    try_match = re.match(r"\s*try\s+(.*)", stem)
    if try_match:
        return tacticTakesHypArgs(try_match.group(1))
    repeat_match = re.match(r"\s*repeat\s+(.*)", stem)
    if repeat_match:
        return tacticTakesHypArgs(repeat_match.group(1))
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
        or stem == "specialize"
    )


def tacticTakesBinderArgs(stem: str) -> bool:
    return stem == "induction"


def tacticTakesIdentifierArg(stem: str) -> bool:
    return stem == "unfold"


def lemma_name_from_statement(stmt: str) -> str:
    if ("Goal" in stmt or "Obligation" in stmt):
        return ""
    stripped_stmt = kill_comments(stmt).strip()
    derive_match = re.fullmatch(
        r"\s*Derive\s+([\w'_]+)\s+SuchThat\s+(.*)\s+As\s+([\w']+)\.\s*",
        stripped_stmt, flags=re.DOTALL)
    if derive_match:
        return derive_match.group(3)
    lemma_match = re.match(
        r"\s*(?:" + "|".join(normal_lemma_starting_patterns) +
        r")\s+([\w'\.]*)(.*)",
        stripped_stmt,
        flags=re.DOTALL)
    assert lemma_match, stripped_stmt
    lemma_name = lemma_match.group(1)
    assert ":" not in lemma_name, stripped_stmt
    return lemma_name


def get_binder_var(goal: str, binder_idx: int) -> Optional[str]:
    paren_depth = 0
    binders_passed = 0
    skip = False
    forall_match = re.match(r"forall\s+", goal.strip())
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


def normalizeNumericArgs(datum: ScrapedTactic) -> ScrapedTactic:
    numerical_induction_match = re.match(
        r"\s*(induction|destruct)\s+(\d+)\s*\.",
        kill_comments(datum.tactic).strip())
    if numerical_induction_match:
        stem = numerical_induction_match.group(1)
        binder_idx = int(numerical_induction_match.group(2))
        binder_var = get_binder_var(datum.context.fg_goals[0].goal, binder_idx)
        if binder_var:
            newtac = stem + " " + binder_var + "."
            return ScrapedTactic(datum.prev_tactics,
                                 datum.relevant_lemmas,
                                 datum.context, newtac)
        else:
            return datum
    else:
        return datum


def parsePPSubgoal(substr: str) -> Obligation:
    split = re.split("\n====+\n", substr)
    assert len(split) == 2, substr
    hypsstr, goal = split
    return Obligation(parse_hyps(hypsstr), goal)


def summarizeContext(context: ProofContext) -> None:
    eprint("Foreground:")
    for i, subgoal in enumerate(context.fg_goals):
        hyps_str = ",".join(get_first_var_in_hyp(hyp)
                            for hyp in subgoal.hypotheses)
        goal_str = re.sub("\n", "\\n", subgoal.goal)[:100]
        eprint(f"S{i}: {hyps_str} -> {goal_str}")


def isValidCommand(command: str) -> bool:
    command = kill_comments(command)
    goal_selector_match = re.fullmatch(r"\s*\d+\s*:(.*)", command,
                                       flags=re.DOTALL)
    if goal_selector_match:
        return isValidCommand(goal_selector_match.group(1))
    return ((command.strip()[-1] == "."
             and not re.match(r"\s*{", command))
            or re.fullmatch(r"\s*[-+*{}]*\s*", command) is not None) \
        and (command.count('(') == command.count(')'))


def load_commands_preserve(args: argparse.Namespace, file_idx: int,
                           filename: str) -> List[str]:
    with open(filename, 'r') as fin:
        contents = fin.read()
    return read_commands_preserve(args, file_idx, contents)


def read_commands_preserve(args: argparse.Namespace, file_idx: int,
                           contents: str) -> List[str]:
    result: List[str] = []
    cur_command = ""
    comment_depth = 0
    in_quote = False
    curPos = 0

    def search_pat(pat: Pattern) -> Tuple[Optional[Match], int]:
        match = pat.search(contents, curPos)
        return match, match.end() if match else len(contents) + 1
    try:
        should_show = args.progress
    except AttributeError:
        should_show = False
    try:
        should_show = should_show or args.read_progress
    except AttributeError:
        pass

    try:
        command_limit = args.command_limit
    except AttributeError:
        command_limit = None

    with tqdm(total=len(contents)+1, file=sys.stdout,
              disable=(not should_show),
              position=(file_idx * 2),
              desc="Reading file", leave=False,
              dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        while curPos < len(contents) and (command_limit is None or
                                          len(result) < command_limit):
            _, next_quote = search_pat(re.compile(r"(?<!\\)\""))
            _, next_open_comment = search_pat(re.compile(r"\(\*"))
            _, next_close_comment = search_pat(re.compile(r"\*\)"))
            _, next_bracket = search_pat(re.compile(r"[\{\}]"))
            next_bullet_match, next_bullet = search_pat(
                re.compile(r"[\+\-\*]+(?![\)\+\-\*])"))
            _, next_period = search_pat(
                re.compile(r"(?<!\.)\.($|\s)|\.\.\.($|\s)"))
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
                if not in_quote and comment_depth > 0:
                    comment_depth -= 1
            elif nextPos == next_bracket:
                if not in_quote and comment_depth == 0 and \
                   re.match(r"\s*(?:\d+\s*:)?\s*$",
                            kill_comments(cur_command[:-1])):
                    result.append(cur_command)
                    cur_command = ""
            elif nextPos == next_bullet:
                assert next_bullet_match
                match_length = next_bullet_match.end() - \
                    next_bullet_match.start()
                if not in_quote and comment_depth == 0 and \
                   re.match(r"\s*$",
                            kill_comments(cur_command[:-match_length])):
                    result.append(cur_command)
                    cur_command = ""
                assert next_bullet_match.end() >= nextPos
            elif nextPos == next_period:
                if not in_quote and comment_depth == 0:
                    result.append(cur_command)
                    cur_command = ""
            curPos = nextPos
    return result


def try_load_lin(args: argparse.Namespace, file_idx: int, filename: str) \
        -> Optional[List[str]]:
    lin_path = Path2(filename + ".lin")
    if args.verbose:
        eprint("Attempting to load cached linearized version from {}"
               .format(lin_path))
    if not lin_path.exists():
        return None
    try:
        ignore_lin_hash = args.ignore_lin_hash
    except AttributeError:
        ignore_lin_hash = False

    with lin_path.open(mode='r') as f:
        first_line = f.readline().strip()
        if ignore_lin_hash or hash_file(filename) == first_line:
            return read_commands_preserve(args, file_idx, f.read())
        else:
            return None


def save_lin(commands: List[str], filename: str) -> None:
    output_file = filename + '.lin'
    with open(output_file, 'w') as f:
        print(hash_file(filename), file=f)
        for command in commands:
            print(command, file=f)


parsePat = re.compile("[() ]", flags=(re.ASCII | re.IGNORECASE))


def parseSexpOneLevel(sexp_str: str) -> Union[List[str], int, Symbol]:
    if sexp_str[0] == '(':
        result = rust_parse_sexp_one_level(sexp_str)
        # eprint(f"Parsing {sexp_str} to {result}")
        return result
    elif re.fullmatch(r"\s*\d+\s*", sexp_str):
        return int(sexp_str.strip())
    elif re.fullmatch(r'\s*\w+\s*', sexp_str):
        return Symbol(sexp_str)
    else:
        assert False, f"Couldn't parse {sexp_str}"
    # sexp_str = sexp_str.strip()
    # if sexp_str[0] == '(':
    #     items = []
    #     cur_pos = 1
    #     item_start_pos = 1
    #     paren_level = 0
    #     while True:
    #         next_match = parsePat.search(sexp_str, cur_pos)
    #         if not next_match:
    #             break
    #         cur_pos = next_match.end()
    #         if sexp_str[cur_pos-1] == "(":
    #             paren_level += 1
    #         elif sexp_str[cur_pos-1] == ")":
    #             paren_level -= 1
    #             if paren_level == 0:
    #                 items.append(sexp_str[item_start_pos:cur_pos])
    #                 item_start_pos = cur_pos
    #         else:
    #             assert sexp_str[cur_pos-1] == " "
    #             if paren_level == 0:
    #                 items.append(sexp_str[item_start_pos:cur_pos])
    #                 item_start_pos = cur_pos
    # elif re.fullmatch(r"\d+", sexp_str):
    #     return int(sexp_str)
    # elif re.fullmatch(r"\w+", sexp_str):
    #     return Symbol(sexp_str)
    # else:
    #     assert False, f"Couldn't parse {sexp_str}"
    # return items


def searchStrsInMsg(sexp, fuel: int = 30) -> List[str]:
    if isinstance(sexp, list) and len(sexp) > 0 and fuel > 0:
        if sexp[0] == "str" or sexp[0] == Symbol("str"):
            assert len(sexp) == 2 and isinstance(sexp[1], str)
            return [sexp[1]]
        else:
            return [substr
                    for substrs in [searchStrsInMsg(sublst, fuel - 1)
                                    for sublst in sexp]
                    for substr in substrs]
    return []


def get_module_from_filename(filename: Union[Path2, str]) -> str:
    return Path2(filename).stem


def symbol_matches(full_symbol: str, shorthand_symbol: str) -> bool:
    if full_symbol == shorthand_symbol:
        return True
    else:
        return full_symbol.split(".")[-1] == shorthand_symbol
    pass


def subgoalSurjective(newsub: Obligation, oldsub: Obligation) -> bool:
    oldhyp_terms = [get_hyp_type(hyp) for hyp in oldsub.hypotheses]
    for newhyp_term in [get_hyp_type(hyp) for hyp in newsub.hypotheses]:
        if newhyp_term not in oldhyp_terms:
            return False
    return newsub.goal == oldsub.goal


def contextSurjective(newcontext: ProofContext, oldcontext: ProofContext):
    for oldsub in oldcontext.all_goals:
        if not any([subgoalSurjective(newsub, oldsub)
                    for newsub in newcontext.all_goals]):
            return False
    return len(newcontext.all_goals) >= len(oldcontext.all_goals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module for interacting with a coq-serapi instance "
        "from Python (3).")
    parser.add_argument(
        "--prelude", default=".", type=str,
        help="The `home` directory in which to look for the _CoqProject file.")
    parser.add_argument(
        "--includes", default=None, type=str,
        help="The include options to pass to coq, as a single string. "
        "If none are provided, we'll attempt to read a _CoqProject "
        "located in the prelude directory, and fall back to no arguments "
        "if none exists.")
    parser.add_argument(
        "--sertop", default="sertop",
        dest="sertopbin", type=str,
        help="The location of the serapi (sertop) binary to use.")
    parser.add_argument(
        "--srcfile", "-f", nargs='*', dest='srcfiles', default=[], type=str,
        help="Coq source file(s) to execute.")
    parser.add_argument(
        "--interactive", "-i",
        action='store_const', const=True, default=False,
        help="Drop into a pdb prompt after executing source file(s). "
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
    with SerapiContext([args.sertopbin],
                       "",
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

import subprocess
import threading
import queue
import sys
import time
import re

class Coq(threading.Thread):
 
    def __init__(self, coq_command):
        threading.Thread.__init__(self, daemon=True)
        self.debug = False
        self.print_errors = True
        self.pdump = None

        self._coqproc = subprocess.Popen(coq_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._fd = self._coqproc.stdout
        self._fin = self._coqproc.stdin
        self._queue = queue.Queue()
        self._last_command = "<no command run>"

    # These functions should be your primary method of interacting with the coq object

    def get_welcome(self):
        return self._queue.get().decode("utf-8")

    def run_command(self, command):
        self._last_command = command
        if self.debug:
            print("Running command: {}".format(command))
        self._fin.write((command.replace('\n', ' ') + '\n').encode('utf-8'))
        self._fin.flush()

    def get(self):
        self.wait_for_prompt()
        # Unfortunately this appears to be necessary... it sometimes
        # takes coqtop a bit to finish printing output, even after
        # it's given you the new prompt.
        time.sleep(0.0005)
        response = self.unsafe_get()
        if self.debug:
            print("Got response:\n{}".format(response.strip()))
        success = check_for_errors(response, self.print_errors)
        if not success:
            if self.print_errors:
                print("Got an error when running {}".format(self._last_command), file=sys.stderr)
            return response, False
        return response, True

    def eget(self):
        response, success = self.get()
        if not success:
            raise CoqError()
        return response

    def kill(self):
        self._coqproc.terminate()
        self._coqproc.stdout.close()
        threading.Thread.join(self)

    # Use these if you're doing something weird

    def wait_for_prompt(self):
        prompt = ""
        while (not " < " in prompt):
            next_prompt_char = self._coqproc.stderr.read(1).decode("utf-8")
            prompt += next_prompt_char
        if self.pdump:
            with open(self.pdump, 'a') as pdump:
                pdump.write(prompt + "\n")

    def unsafe_get(self):
        result = ""
        while not self._queue.empty():
            line = self._queue.get().decode("utf-8")
            result = result + line + "\n"
        return result

    # Internal
    def run(self):
        '''The body of the thread: read lines and put them on the queue.'''
        while(True):
            try:
                line = self._fd.readline()
                self._queue.put(line)
            except:
                break

class CoqError(Exception):
    def __str__(self):
        return "CoqError"

def check_for_errors(response, print_message=True):
    success = True
    for line in response.split("\n"):
        if "Error:" in line:
            if success and print_message:
                print("Got error:")
            success = False
        if not success and print_message:
            print(line, file=sys.stderr)
    return success

def has_toplevel_colonequals(command):
    depth = 0
    for i in range(len(command)):
        if re.match("\slet\s", command[i:i+5]):
            depth += 1
        if re.match("\sin\s", command[i:i+4]):
            depth -= 1
        if re.match(":=\s", command[i:i+4]) and depth == 0:
            return True
    return False

def starting_proof(command):
    return ((re.match("Lemma\s", command) or
             re.match("Theorem\s", command) or
             re.match("Remark\s", command) or
             re.match("Proposition\s", command) or
             re.match("Definition\s", command) or
             re.match("Example\s", command) or
             re.match("Fixpoint\s", command) or
             re.match("Corollary\s", command) or
             ("Instance" in command and
              "Declare" not in command)) and
            not has_toplevel_colonequals(command))

def ending_proof(command):
    return ("Qed" in command or
            "Defined" in command or
            (re.match("Proof\s+\S+\s*", command) and
             not re.match("Proof with", command)))

def get_goal(response):
    inGoal = False
    goal = ""
    for line in response.split("\n"):
        if "subgoal" not in line and inGoal and line != "":
            goal += line.strip() + "\n"
        elif "=====" in line:
            inGoal = True
        elif "is:" in line:
            inGoal = False
    return goal

def get_context(response):
    inContext = False
    context = ""
    for line in response.split("\n"):
        if "=====" in line:
            inContext = False
        if inContext and line.strip() != "":
            context += line.strip() + "\n"
        if "subgoal" in line:
            inContext = True
    return context

def split_compound_command(command):
    # The "now" tactic modifier associates weirdly.
    if (re.match("now\s+\S+", command)):
        return [command.strip()]
    cur_command = ""
    commands = []
    depth = 0
    for ch in command:
        if ch == '[' or ch == '(':
            depth = depth + 1
        elif ch == ']' or ch == ')':
            depth = depth - 1
        if ch == ';' and depth == 0:
            commands.append(cur_command.strip())
            cur_command = ""
        else:
            cur_command = cur_command + ch
    commands.append(cur_command.strip())
    return commands

def kill_comments(string):
    result = ""
    depth = 0
    for i in range(len(string)):
        if string[i:i+2] == '(*':
            depth += 1
        if depth == 0:
            result += string[i]
        if string[i-1:i+1] == '*)':
            depth -= 1
    return result

def kill_brackets(string):
    result = re.sub(re.compile("(\.\s*)\}", re.DOTALL), r'\1', string)
    result = re.sub(re.compile("(\.\s*)\{", re.DOTALL), r'\1', result)
    return result

def kill_bullets(string):
    result = re.sub(re.compile("(\.\s*)[+*-]", re.DOTALL), r'\1', string)
    return result

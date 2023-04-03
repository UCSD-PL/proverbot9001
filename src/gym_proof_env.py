# Gym interface for coqproofenv

import multiprocessing
import torch
import wandb
import time, random
import re
import coq_serapy
from coq_serapy import (load_commands, kill_comments, split_tactic,
                        contextSurjective, ending_proof, admit_proof)
from coq_serapy.contexts import (truncate_tactic_context, FullContext,
                                 assert_proof_context_matches)
from search_file import loadPredictorByFile
from search_strategies import completed_proof
import gym
import os, sys
from util import nostderr, unwrap, eprint, mybarfmt
from multiprocessing import Pipe, Process
import io
import coq2vec
import gymnasium as gym
from gymnasium.spaces import Discrete

class ActionSpace:
    def __init__(self,ls_actions):
        self.ls_actions = ls_actions
        self.length = len(ls_actions)
    def sample(self):
        if self.length == 0:
            return -1
        else:
            return random.randrange(0,self.length)
    def get_action_by_index(self,idx):
        if idx<0:
            return None
        else:
            return self.ls_actions[idx]

class ProofEnv(gym.Env):
    def __init__(self, proof_files, prelude, wandb = False, time_per_command=100, max_proof_len = 50, write_solved_proofs = True,
                                state_type = "index"):
        self.action_space = None
        self.observation_space = None
        self.prelude= prelude
        self.proof_files = proof_files
        self.proof_file_index = 0
        self.proof_line_num = 0
        self.wandb_log = wandb
        self.coq: Optional[SerapiInstance] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.write_solved_proofs = write_solved_proofs

        # TODO: see if we can put predictor in the envionment?
        self.time_per_command= time_per_command
        self.max_proof_len = max_proof_len
        os.makedirs("output", exist_ok=True)
        self.proof_lines_file = "output/output_proof_lines.txt"
        with open(self.proof_lines_file,"w") as f :
            f.write("")

        self.max_num_proofs = 15
        self.num_proofs = 0
        self._load_list_tactic_classes()
    def _prooflines_file_write(self, write_str) :
        if self.write_solved_proofs :
            with open(self.proof_lines_file,"a") as f :
                f.write(write_str)
                f.flush()

    def _load_list_tactic_classes(self) :
        with open("tactics.txt", "r") as f :
            whole_file = f.read()
            self.list_of_tactic_classes = whole_file.split("\n")
            for i in range(len(self.list_of_tactic_classes)) :
                self.list_of_tactic_classes[i] = self.list_of_tactic_classes[i].strip().rstrip(".")

    def run_to_proof(self, proof_contains) :
        print("Running to proof",proof_contains)
        while self.proof_line_num < len(self.commands) :# and  self.num_proofs <= self.max_num_proofs :
            if proof_contains in self.commands[self.proof_line_num] :
                print("Found Proof : ", kill_comments(self.commands[self.proof_line_num].lstrip().rstrip()))

                self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)

                self.proof_line_num += 1
                self.num_proofs  += 1
                self.proof_contexts_in_path.append(self.coq.proof_context)
                break
            else:
                not_function = kill_comments(self.commands[self.proof_line_num - 1]).lstrip().rstrip().split()[0].lower() != "function"
                if self.commands[self.proof_line_num].lstrip().rstrip() == "Proof." and not_function:
                    self.num_proofs  += 1
                self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
                self.proof_line_num += 1

    def _goto_next_proof(self):
        assert self.coq.proof_context is None
        self.end_proof_time = time.time()
        self.num_commands = 0
        self.proof_contexts_in_path = []
        while self.proof_line_num < len(self.commands) :# and  self.num_proofs <= self.max_num_proofs :
            not_function = kill_comments(self.commands[self.proof_line_num - 1]).lstrip().rstrip().split()[0].lower() != "function"
            if self.commands[self.proof_line_num].lstrip().rstrip() == "Proof." and not_function:
                print(self.commands[self.proof_line_num - 1].lstrip().rstrip().split()[0].lower())

                self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
                self.proof_line_num += 1
                self.num_proofs  += 1
                self.proof_contexts_in_path.append(self.coq.proof_context)
                break
            else :
                self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
                self.proof_line_num += 1
        if self.proof_line_num >= len(self.commands) : #or self.num_proofs >= self.max_num_proofs :
            print("File Finished")
            self._reset_to_start_of_file()

            return self._goto_next_proof()

        self.proof_time = self.end_proof_time - self.start_proof_time
        self.start_proof_time = time.time()
        self.proof_time_calculated = sum(self.debug_time)
        self.debug_time = []
        return None #self.get_state_vector( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip())


    def _navigate_file_end_of_current_proof(self) :
        # This is wrong - Not all proofs end with QED. Also make this section cleaner.
        print("Navigating file to the end of current proof without running")
        assert self.coq.proof_context is None
        while self.proof_line_num < len(self.commands)  and not ending_proof(self.commands[self.proof_line_num]) :
            print("Navigating ", self.commands[self.proof_line_num])
            self.proof_line_num += 1
        print("Navigating finished", self.commands[self.proof_line_num], ending_proof(self.commands[self.proof_line_num]))
        self.proof_line_num += 1
        # print("Navigated to :", self.commands[self.proof_line_num] )

    def _reset_to_start_of_file(self) :
        if self.coq is not None:
            self.coq.kill()
        self.coq = coq_serapy.SerapiInstance(['sertop', '--implicit'],None, prelude = self.prelude)
        self.coq.verbose = 3
        self.coq.quiet = True
        self.proof_line_num = 0
        self.num_proofs = 0
        self.num_proofs_solved = 0
        self.commands = load_commands(self.proof_files[self.proof_file_index], progress_bar=True)
        print("Starting File :", self.proof_files[self.proof_file_index])

        self.proof_file_index = (self.proof_file_index + 1) % len(self.proof_files)

    def _admit_and_skip_proof(self) :
        self._prooflines_file_write("\n".join(self.coq.tactic_history.getFullHistory()))
        ending_command = None
        for cmd in self.commands[self.proof_line_num:]:
            if ending_proof(cmd):
                ending_command = cmd
                break
        assert ending_command
        lemma_statement = self.coq.prev_tactics[0]

        proof_relevant = ending_command.strip() == "Defined." or \
            bool(re.match(
                r"\s*Derive",
                kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Let",
                kill_comments(lemma_statement))) or \
            bool(re.match(
                r"\s*Equations",
                kill_comments(lemma_statement)))
        if proof_relevant:
            while len(self.coq.prev_tactics) > 1:
                self.coq.cancel_last()
            _, run_cmds = unwrap(self.coq.finish_proof(
                self.commands[self.proof_line_num:]))
            self.proof_line_num += len(run_cmds)
            self._prooflines_file_write("\n".join(run_cmds))
        else:
            try:
                admitting_cmds = admit_proof(self.coq, lemma_statement, ending_command)
                self._prooflines_file_write("\n".join(admitting_cmds))
            except coq_serapy.SerapiException:
                lemma_name = \
                  coq_serapy.lemma_name_from_statement(lemma_statement)
                eprint(f"{self.cur_file}: Failed to admit proof {lemma_name}")
                raise
            while not coq_serapy.ending_proof(self.commands[self.proof_line_num]):
                self.proof_line_num += 1
            self.proof_line_num += 1
        if self.wandb_log :
            wandb.log({"Num command Attempts" : self.num_commands  })
        self._goto_next_proof()
        done = True
        next_state = self.coq.proof_context
        r = 0#-1
        info = {}
        info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
        info["next_state"] = next_state
        return next_state, r, done, info

    def _is_context_fresh(self, curr_proof_context) :
        # print(len(self.proof_contexts_in_path))
        for context in self.proof_contexts_in_path :
            if contextSurjective(curr_proof_context, context) :
                return False
        return True

    def _check_next_state(self,prediction):
        info = {}
        eprint("Checking next state for action -", prediction)
        tactic_class,tactic_args = split_tactic(prediction.strip().rstrip("."))
        if tactic_class.lower().strip() == "exploit" :
            return [],info
        next_state = []
        context_before = self.coq.proof_context
        a= time.time()
        try:
            self.coq.run_stmt(prediction, timeout= self.time_per_command)

        except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
                coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
                coq_serapy.ParseError,
                RecursionError,
                coq_serapy.UnrecognizedError) as e:
            eprint("One of known errors", e)
        except coq_serapy.CoqAnomaly:
            eprint("Coq Anomaly")
            # self.kill()
            quit()
        else :
            b = time.time()
            eprint("Time for running the above command", b-a)
            num_brackets_run = 0
            while len(unwrap(self.coq.proof_context).fg_goals) == 0 and not completed_proof(self.coq):
                if len(unwrap(self.coq.proof_context).shelved_goals) > 0:
                    print("Running Unshelve.")
                    self.coq.run_stmt("Unshelve.", timeout= self.time_per_command)
                    num_brackets_run += 1
                    continue
                print("Running }")
                self.coq.run_stmt("}", timeout= self.time_per_command)
                num_brackets_run += 1

            if len(self.coq.proof_context.fg_goals) > 1 :
                print("Context before running open brace :",self.coq.proof_context)
                print("Running {")
                self.coq.run_stmt( "{", timeout= self.time_per_command)
                eprint("Context after running open brace :",self.coq.proof_context)
                num_brackets_run += 1

            if completed_proof(self.coq) :
                next_state = self.coq.proof_context
                for _ in range(num_brackets_run) :
                    self.coq.cancel_last()
                self.coq.cancel_last()
                # print("QED on this action. Cancelled - ",prediction)
                info["state_text"] = "fin"
                return next_state,info

            if self.coq.proof_context == None :
                print("Something is wrong. Lost context")
                quit()

            if self._is_context_fresh(self.coq.proof_context) :
                next_state_name =  self.coq.proof_context.fg_goals[0].goal
                next_state =self.coq.proof_context
                info["state_text"] = next_state_name.strip()
                eprint("Context is fresh for this actions")
            else :
                eprint("Context is not fresh for this action")
                next_state = []

            if num_brackets_run > 0 :
                eprint("Cancelling", num_brackets_run, "Brackets")
                for _ in range(num_brackets_run) :
                    self.coq.cancel_last(force_update_nonfg_goals=True)

            context_mid = self.coq.proof_context
            self.coq.cancel_last(force_update_nonfg_goals=True)
            eprint("Cancelled - ",prediction)

        context_after = self.coq.proof_context

        assert_proof_context_matches(context_before, context_after)
        return next_state,info


    def step(self, action=None):
        """
        Run one timestep of the environment's dynamics using the agent actions.
            When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
            reset this environment's state for the next episode.
            .. versionchanged:: 0.26
                The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
                to users when the environment had terminated or truncated which is critical for reinforcement learning
                bootstrapping algorithms.
            Args:
                action (ActType): an action provided by the agent to update the environment state.
            Returns:
                observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                    An example is a numpy array containing the positions and velocities of the pole in CartPole.
                reward (SupportsFloat): The reward as a result of taking the action.
                terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                    which can be positive or negative. An example is reaching the goal state or moving into the lava from
                    the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
                truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                    Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                    Can be used to end the episode prematurely before a terminal state is reached.
                    If true, the user needs to call :meth:`reset`.
                info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                    This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                    hidden from observations, or individual reward terms that are combined to produce the total reward.
                    In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                    however this is deprecated in favour of returning terminated and truncated variables.
                done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                    return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                    A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                    a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """

        if action is None:
            r = 0
            s_next,episode_r, done, info = self._admit_and_skip_proof()
            return s_next,episode_r, done, info # If done, we no longer include next-states etc. in info
        done = False
        # prediction = self.get_pred(action)
        prediction = action.prediction
        info = {}
        eprint("Taking step -", action)
        a= time.time()
        try:
            self.coq.run_stmt(prediction, timeout= self.time_per_command, force_update_nonfg_goals=True)
        except (coq_serapy.CoqTimeoutError, coq_serapy.ParseError,
                coq_serapy.CoqExn, coq_serapy.CoqOverflowError,
                coq_serapy.ParseError,
                RecursionError,
                coq_serapy.UnrecognizedError) as e:
            eprint("One of known errors", e)
            r = 0
            s_next,episode_r, done, info = self._admit_and_skip_proof()
            return s_next,episode_r, done, info # If done, we no longer include next-states etc. in info
        except coq_serapy.CoqAnomaly:
            eprint("Coq Anomaly")
            # self.kill()
            quit()
        else :
            b = time.time()
            self.debug_time.append(b-a)
            print("Time for running the above command", b-a)
            r = 0 #No rewards for progress
            a = time.time()
            while len(unwrap(self.coq.proof_context).fg_goals) == 0 and not completed_proof(self.coq):
                if len(unwrap(self.coq.proof_context).shelved_goals) > 0:
                    print("Running Unshelve.")
                    self.coq.run_stmt("Unshelve.", timeout= self.time_per_command)
                    continue
                print("Running }")
                self.coq.run_stmt("}", timeout= self.time_per_command)
            b = time.time()
            self.debug_time.append(b-a)

            a = time.time()
            if len(self.coq.proof_context.fg_goals) > 1 :
                print(self.coq.proof_context.fg_goals,self.coq.proof_context.bg_goals)
                print("Running {")
                self.coq.run_stmt( "{", timeout= self.time_per_command)
            b = time.time()
            self.debug_time.append(b-a)

            a= time.time()
            if completed_proof(self.coq) :
                if self.wandb_log :
                    wandb.log({"Num command Attempts" : self.num_commands  })
                b = time.time()
                self.debug_time.append(b-a)
                tactics = self.coq.tactic_history.getFullHistory() + ["Qed."]
                self.coq.run_stmt( "Qed.", timeout= self.time_per_command)
                r = 1
                print("Current proof fin with Good rewards")
                self._prooflines_file_write("\n".join(tactics))
                self.num_proofs_solved += 1
                a = time.time()
                self._navigate_file_end_of_current_proof()
                b = time.time()
                self.debug_time.append(b-a)
                print("Time taken to naviagate file to the end of current proof", b -a)
                a = time.time()
                self._goto_next_proof()
                b = time.time()
                self.debug_time.append(b-a)
                print("Time taken to run goto_next_proof function", b-a)
                done = True
                next_state = self.coq.proof_context
                info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
                return next_state, r, done, info
            b = time.time()
            self.debug_time.append(b-a)
            print("Time taken to check completed proof", b - a)

            if self.coq.proof_context == None :
                print("No context")
                quit()

            self.num_commands += 1
            a = time.time()
            assert self._is_context_fresh(self.coq.proof_context)
            b = time.time()
            self.debug_time.append(b-a)
            print("Time taken to check if context fresh", b - a)
            self.proof_contexts_in_path.append(self.coq.proof_context)


        if self.num_commands > self.max_proof_len :
            a = time.time()
            result = self._admit_and_skip_proof()
            b = time.time()
            self.debug_time.append(b-a)
            print("Time taken to run admit and skip proof", b-a)
            return result
        next_state = self.coq.proof_context
        info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
        return next_state, r, done, info  #next_obs, rewards, dones, infos

    def reset(self):
        self._reset_to_start_of_file()
        self.start_proof_time = 0
        self.debug_time = []
        self._goto_next_proof()
        # state = self.get_state_vector( self.coq.proof_context )
        state = self.coq.proof_context
        info = {}
        info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
        info["next_states"] = state
        return (state,info)

def child_process(pid, critical, pipe) :
    import sys
    os.makedirs("output/results", exist_ok=True)
    os.makedirs("output/errors", exist_ok=True)
    open("output/results/subprocess_pid%d_out.txt"%pid, 'w').close()
    open("output/errors/subprocess_pid%d_error.txt"%pid, 'w').close()
    proof_file, prelude, time_per_command, state_type, max_proof_len = critical
    test_env = ProofEnv(proof_file, prelude, wandb = False, time_per_command = time_per_command, write_solved_proofs=False, max_proof_len=max_proof_len, state_type = state_type)
    sys.stdout = open("output/results/subprocess_pid%d_out.txt"%pid, 'a')
    sys.stderr = open("output/errors/subprocess_pid%d_error.txt"%pid, 'a')
    print("child process created", pid)
    while True :
        if pipe.poll(1800) :
            func, args = pipe.recv()
        else :
            print("Terminating Child process", pid, "due to timeout")
            quit()
        print("This is inside a given child -", func, args)
        if func == 'reset' :
            result = test_env.reset()
        elif func == 'step' :
            result = test_env.step(args)
        elif func == 'check_next_state' :
            result = test_env._check_next_state(args)
            print("Results of Check next state inside a child - ",result)
        elif func == 'admit_and_skip_proof' :
            result = test_env._admit_and_skip_proof()
        elif func == 'run_to_proof' :
            result = test_env.run_to_proof(args)
        elif func == 'terminate' :
            break
        elif func == 'keepalive' :
            result = ""
        else :
            raise ValueError("Unknown function")

        pipe.send(result)

    return


class FastProofEnv(gym.Env):
    def __init__(self, proof_file, prelude, wandb = False, time_per_command=100, write_solved_proofs = True, state_type = "vector",
                    max_proof_len = 30, num_check_engines = 5, weightsfile="data/polyarg-weights.dat",max_term_length=256):
        self.proof_file = proof_file
        self.action_space = None
        self.prelude = prelude
        self.wandb = wandb
        self.time_per_command = time_per_command
        self.state_type = state_type
        self.num_check_engines = num_check_engines
        self.max_proof_len = max_proof_len
        self.main_engine = ProofEnv(proof_file, prelude, wandb, time_per_command, write_solved_proofs = write_solved_proofs, max_proof_len=max_proof_len, state_type = state_type)
        print("weightsfile: ",weightsfile)
        self.predictor = loadPredictorByFile(weightsfile)
        self._create_pipes_and_children()
        self.max_term_length = max_term_length
        self.state_encoder = self._get_state_encoder()

    def _get_state_encoder(self):
        termvectorizer = coq2vec.CoqTermRNNVectorizer()
        termvectorizer.load_weights("data/term2vec-weights-59.dat")
        return termvectorizer

    def _encode_state(self,state):
        with torch.no_grad():
            if len(state.fg_goals) > 0:
                s_enc = self.state_encoder.term_to_vector(state.fg_goals[0].goal).flatten()
            else:
                s_enc = self.state_encoder.term_to_vector("").flatten()
        return s_enc
    def _encode_next_states(self,ls_states):
        if len(ls_states)>0:
            encoded_next_states = torch.stack([self._encode_state(state) for state in ls_states], dim=0)
        else:
            encoded_next_states = torch.tensor([])
        return encoded_next_states
    @property
    def coq(self):
        return self.main_engine.coq
    @property
    def num_proofs(self) :
        return self.main_engine.num_proofs
    @property
    def num_proofs_solved(self) :
        return self.main_engine.num_proofs_solved
    @property
    def proof_time(self) :
        return self.main_engine.proof_time
    @property
    def proof_time_calculated(self) :
        return self.main_engine.proof_time_calculated
    @property
    def curr_proof_tactics(self):
        return self.coq.tactic_history.getFullHistory()
    def _create_pipes_and_children(self) :
        self.server_end_pipes = []
        self.child_end_pipes = []
        for i in range(self.num_check_engines) :
            s,c = Pipe()
            self.server_end_pipes.append(s)
            self.child_end_pipes.append(c)
        # print(self.num_check_engines)
        process_list = []
        context = multiprocessing.get_context('fork')
        for i in range(self.num_check_engines) :
            p = context.Process(target=child_process, args=(i,(self.proof_file, self.prelude,
                self.time_per_command, self.state_type, self.max_proof_len),self.child_end_pipes[i] ) )
            p.start()
            process_list.append(p)

        print("Exploratory Environments successfully running")
        return

    def _admit_and_skip_proof(self):
        print("Admitting and Skipping the current proof on all Test engines")
        for pipe in self.server_end_pipes :
            pipe.send( ["admit_and_skip_proof",None])
        for pipe in self.server_end_pipes :
            pipe.recv()
        print("Test engines sucessfully skipped proof")
        return self.main_engine._admit_and_skip_proof()

    def reset(self) :
        state,info = self.main_engine.reset()
        for pipe in self.server_end_pipes :
            pipe.send(["reset",None])
        for pipe in self.server_end_pipes :
            pipe.recv()
        print("All Test Engines Reset")
        state_encoded = self._encode_state(state)
        next_states, list_of_pred, next_state_texts = self._get_available_actions_with_next_state_vectors()
        # quit()
        next_states_encoded = self._encode_next_states(next_states)
        info['reachable_states'] = next_states_encoded
        info['list_of_pred'] = list_of_pred
        info['reachable_states_text'] = next_state_texts
        self.action_space = ActionSpace(list_of_pred)
        return state_encoded,info

    def step(self, action) :
        print("Stepping on all Test Engines")
        for pipe in self.server_end_pipes :
            pipe.send(["step",action])
        for pipe in self.server_end_pipes :
            pipe.recv()
        s_next,episode_r, done, info = self.main_engine.step(action)
        s_next_encoded = self._encode_state(s_next)
        next_states, list_of_pred, next_state_texts = self._get_available_actions_with_next_state_vectors()
        next_states_encoded = self._encode_next_states(next_states)
        # self.num_proofs = self.main_engine.num_proofs
        # self.num_proofs_solved = self.main_engine.num_proofs_solved
        info['reachable_states'] = next_states_encoded
        info['list_of_pred'] = list_of_pred
        info['reachable_states_text'] = next_state_texts
        self.action_space = ActionSpace(list_of_pred)
        return s_next_encoded, episode_r, done, info

    def _check_next_states(self,predictions):
        print("Checking next States on all Test Engines")
        a = time.time()
        for i in range(len(predictions)) :
            self.server_end_pipes[i].send(["check_next_state",predictions[i]])
        results = []
        for pipe in self.server_end_pipes :
            recv_obj = pipe.recv()
            results.append(recv_obj)
        b = time.time()
        # print(results)
        print("Time for check next states", b - a)
        # quit()
        return list(zip(*results))

    def run_to_proof(self, proof_contains) :
        print("Running to proof on all Test states")
        for pipe in self.server_end_pipes :
            pipe.send(["run_to_proof",proof_contains])
        for pipe in self.server_end_pipes :
            pipe.recv()
        return self.main_engine.run_to_proof(proof_contains)

    def _get_available_actions_with_next_state_vectors(self) :
        relevant_lemmas = self.coq.local_lemmas[:-1]
        # print(self.coq.proof_context)
        full_context_before = FullContext(relevant_lemmas, self.coq.prev_tactics,  self.coq.proof_context)
        predictions = self.predictor.predictKTactics(
            truncate_tactic_context(full_context_before.as_tcontext(),
                                    self.max_term_length), self.num_check_engines)
        next_states = []
        list_of_pred = []
        next_state_texts = []
        print("Available actions", [_.prediction for _ in predictions])
        all_available_pred =  [_.prediction.lstrip().rstrip() for _ in predictions]
        result = self._check_next_states(all_available_pred)
        # print(result)
        # quit()
        all_next_states, all_next_infos = result
        # print(all_next_states)
        # print(all_next_infos)
        for next_state_ind in range(len(all_next_states)) :
            curr_next_state = all_next_states[next_state_ind]
            if len(curr_next_state) == 0 or repeating_actions(predictions[next_state_ind], self.coq.prev_tactics):
                continue
            else :
                curr_next_state_text = all_next_infos[next_state_ind]["state_text"]
                next_states.append(curr_next_state)
                list_of_pred.append(predictions[next_state_ind] )
                next_state_texts.append(curr_next_state_text)
        return next_states, list_of_pred, next_state_texts


def repeating_actions(action, tactics_used, cutoff = 6) :
    if len(tactics_used) < cutoff :
        return False
    if tactics_used[-cutoff:].count(tactics_used[-1]) == cutoff :
        return True
    return False

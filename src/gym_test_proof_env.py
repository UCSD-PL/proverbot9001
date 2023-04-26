import multiprocessing
import time
import random
import re
import os
import sys
from pathlib import Path
from typing import Optional, List
from multiprocessing import Pipe

import torch
import wandb
import gymnasium as gym
import coq_serapy
import coq2vec
from coq_serapy import (load_commands, kill_comments, split_tactic,
                        contextSurjective, ending_proof, admit_proof, CoqAgent)
from coq_serapy.contexts import (truncate_tactic_context, FullContext,
                                 assert_proof_context_matches)
from dataloader import TacticContext
from models import tactic_predictor

from search_file import loadPredictorByFile
from search_strategies import completed_proof
from search_worker import ReportJob
from util import unwrap, eprint
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
    def __getitem__(self,idx):
        if idx<0:
            return None
        else:
            return self.ls_actions[idx]


class DummyPredictor():
    def __init__(self) -> None:
        pass

    def predictKTactics(self, in_data : TacticContext, k : int,
                        blacklist: Optional[List[str]] = None) \
        -> List[tactic_predictor.Prediction]:
        del blacklist
        del in_data
        del k
        return [tactic_predictor.Prediction("intro.", 0.25), tactic_predictor.Prediction("apply conj.", 0.25),
                tactic_predictor.Prediction("reflexivity.", 0.25), tactic_predictor.Prediction("simpl.", 0.25)]


class TestProofEnv :
    def __init__(self) :
        proof1 = [ "Theorem negb_involutive : forall b : bool,  negb (negb b) = b.", "Proof.", "intros b.", "destruct b.", "{", "reflexivity.", "}", "{", "reflexivity.", "}", "Qed."]
        proof2 = [ "Example test_mult1: (mult 3 3) = 9.", "Proof.", "simpl.", "reflexivity.", "Qed."]
        proof1 = ["Theorem thrm: forall n: nat, n = n /\ 1 + n = S n.", "Proof"]

        proof1_actions = { "Proof." : ["intros b.", "destruct b."], 
                           "Proof.intros b." : ["destruct b.{", "intuition."],
                           "Proof.intros b.destruct b.{" : ["reflexivity. } {", "destruct negb."]}
        
        self.current_proof = "Theorem thrm2: forall n: nat, n = n /\ 1 + n = S n."
        self.time_per_command = 100
        self.coq = None
        self.predictor = DummyPredictor()
        self.num_check_engines = 4
        self.max_term_length = 256
        self.prelude = "CompCert"
        self.proof_contexts_in_path = []
        self.proofs_done: List[ReportJob] = []
        self.state_encoder = self._get_state_encoder()
        self.proof_file_index = 0
        self.max_proof_len = 50

    def _get_state_encoder(self):
        termvectorizer = coq2vec.CoqTermRNNVectorizer()
        termvectorizer.load_weights("data/term2vec-weights-59.dat")
        return termvectorizer

    def _is_context_fresh(self, curr_proof_context) :
        # print(len(self.proof_contexts_in_path))
        for context in self.proof_contexts_in_path :
            if contextSurjective(curr_proof_context, context) :
                return False
        return True
    
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
    
    def reset(self) :
        self.proof_contexts_in_path = []
        self.num_commands = 0
        
        if self.coq is not None:
            self.coq.kill()
        self.coq = coq_serapy.SerapiInstance(['sertop', '--implicit'],None, prelude = str(self.prelude))
        self.coq.verbose = 3
        self.coq.quiet = True
        self.coq.run_stmt(self.current_proof, timeout= self.time_per_command, force_update_nonfg_goals=True)
        self.coq.run_stmt("Proof.", timeout= self.time_per_command, force_update_nonfg_goals=True)
        
        state = self.coq.proof_context
        state_encoded = self._encode_state(state)
        next_states, list_of_pred, next_state_texts = self._get_available_actions_with_next_state_vectors()
        # quit()
        next_states_encoded = self._encode_next_states(next_states)
        print("List of allowed Predictions :", list_of_pred)
        self.action_space = ActionSpace(list_of_pred)
        self.reachable_states = next_states_encoded

        #CoqAgent.locate_ident()
        return state_encoded

    def _admit_and_skip_proof(self) :
        print("Admitting and skippping proof")
        self.coq.run_stmt("Admitted.")
        next_state_encoded = self.reset()
        done = 1
        r = 0
        info = {}
        return next_state_encoded, r, done, info


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
    
    def _check_next_states(self,predictions):
        results = []
        for prediction in predictions :
            eprint("Checking next state for action -", prediction)
            next_state = []
            info = {}
            context_before = self.coq.proof_context
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
                    print("QED on this action. Cancelled - ",prediction)
                    info["state_text"] = "fin"
                    results.append( (next_state,info) )
                    continue 

                if self.coq.proof_context == None :
                    print("Something is wrong. Lost context")
                    quit()

                if self._is_context_fresh(self.coq.proof_context) :
                    next_state_name =  self.coq.proof_context.fg_goals[0].goal
                    next_state = self.coq.proof_context
                    info["state_text"] = next_state_name.strip()
                    eprint("Context is fresh for this actions")
                else :
                    eprint("Context is not fresh for this action")
                    next_state = []

                if num_brackets_run > 0 :
                    eprint("Cancelling", num_brackets_run, "Brackets")
                    for _ in range(num_brackets_run) :
                        self.coq.cancel_last(force_update_nonfg_goals=True)

                self.coq.cancel_last(force_update_nonfg_goals=True)
                eprint("Cancelled - ",prediction)

            context_after = self.coq.proof_context
            assert_proof_context_matches(context_before, context_after)
            results.append( (next_state,info) )
   
        return list(zip(*results))
    

    def step(self,action_indx) :
        print("action index", action_indx)
        prediction = self.action_space[action_indx]
        if prediction is None:
            r = 0
            s_next,episode_r, done, info = self._admit_and_skip_proof()
            return s_next,episode_r, done, info
        
        action = prediction.prediction
        info = {}
        done = False
        print("Taking step", action)
        try:
            self.coq.run_stmt(action, timeout= self.time_per_command, force_update_nonfg_goals=True)
        except Exception as e:
            print("action failed :", e)
            r = 0
            s_next,episode_r, done, info = self._admit_and_skip_proof()
            return s_next,episode_r, done, info 
        else :
            r = 0 #No rewards for progress
            while len(unwrap(self.coq.proof_context).fg_goals) == 0 and not completed_proof(self.coq):
                if len(unwrap(self.coq.proof_context).shelved_goals) > 0:
                    print("Running Unshelve.")
                    self.coq.run_stmt("Unshelve.", timeout= self.time_per_command)
                    continue
                print("Running }")
                self.coq.run_stmt("}", timeout= self.time_per_command)
            
            if len(self.coq.proof_context.fg_goals) > 1 :
                print(self.coq.proof_context.fg_goals,self.coq.proof_context.bg_goals)
                print("Running {")
                self.coq.run_stmt( "{", timeout= self.time_per_command)
            
            if completed_proof(self.coq) :
                self.coq.run_stmt( "Qed.", timeout= self.time_per_command)
                r = 1
                print("Current proof fin with Good rewards")
                self.reset()
                done = True
                next_state = self.coq.proof_context
                s_next_encoded = self._encode_state(next_state)
                info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
                return s_next_encoded, r, done, info
            if self.coq.proof_context == None :
                print("No context")
                quit()

            self.num_commands += 1
            assert self._is_context_fresh(self.coq.proof_context)
            self.proof_contexts_in_path.append(self.coq.proof_context)


        if self.num_commands > self.max_proof_len :
            print("Too many actions, skipping proof")
            result = self._admit_and_skip_proof()
            return result
        next_state = self.coq.proof_context
        info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()

        s_next_encoded = self._encode_state(next_state)
        next_states, list_of_pred, next_state_texts = self._get_available_actions_with_next_state_vectors()
        next_states_encoded = self._encode_next_states(next_states)
        # self.num_proofs = self.main_engine.num_proofs
        # self.num_proofs_solved = self.main_engine.num_proofs_solved
        info['reachable_states'] = next_states_encoded
        info['list_of_pred'] = list_of_pred
        info['reachable_states_text'] = next_state_texts
        print("List of allowed Predictions :", list_of_pred)
        self.action_space = ActionSpace(list_of_pred)
        self.reachable_states = next_states_encoded
        return s_next_encoded, r, done, info
    



def repeating_actions(action, tactics_used, cutoff = 6) :
    if len(tactics_used) < cutoff :
        return False
    if tactics_used[-cutoff:].count(tactics_used[-1]) == cutoff :
        return True
    return False